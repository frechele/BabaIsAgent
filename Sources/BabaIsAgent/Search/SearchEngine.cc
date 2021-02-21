#include <BabaIsAgent/Search/SearchEngine.hpp>

#include <BabaIsAgent/Utils/Utils.hpp>

#include <chrono>
#include <effolkronium/random.hpp>
#include <iomanip>
#include <iostream>
#include <random>

namespace BabaIsAgent::Search
{
SearchEngine::SearchEngine(std::string_view configFileName)
    : configFileName_(std::move(configFileName)),
      netMgr_(config_),
      controller_(config_)
{
    loadConfig();

    netMgr_.Init();

    for (int threadId = 0; threadId < config_.NumOfSearchWorker; ++threadId)
    {
        searchWorkers_.emplace_back(&SearchEngine::searchThread, this);
    }

    deleteWorker_ = std::thread(&SearchEngine::deleteThread, this);

    updateRoot(nullptr);
}

SearchEngine::~SearchEngine() noexcept
{
    controller_.Terminate();

    for (auto& worker : searchWorkers_)
    {
        if (worker.joinable())
            worker.join();
    }

    deleteNode(root_);

    // maybe root is deleted during this time.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    isDeleteThreadRunning_ = false;
    if (deleteWorker_.joinable())
        deleteWorker_.join();
}

void SearchEngine::Init(std::string_view gameFileName)
{
    mainGame_.reset(new baba_is_auto::Game(std::move(gameFileName)));
}

void SearchEngine::DoSearch()
{
    loadConfig();
    resumeSearch();

    bool stopFlag = false;
    while (!stopFlag)
    {
        stopFlag |= (numOfSimulations_.load() >= config_.MaxSimulationCount);
    }

    pauseSearch();
}

void SearchEngine::Play(baba_is_auto::Direction action)
{
    pauseSearch();

    mainGame_->MovePlayer(action);

    TreeNode* newRoot = nullptr;
    root_->ForEach([&newRoot, action](TreeNode* child) {
        if (child->Action == action)
        {
            newRoot = child;
        }
    });

    updateRoot(newRoot);
}

void SearchEngine::DumpStats() const
{
    std::vector<const TreeNode*> children;
    root_->ForEach(
        [&children](TreeNode* child) { children.emplace_back(child); });

    std::sort(begin(children), end(children),
              [](const TreeNode* a, const TreeNode* b) {
                  return a->Visits > b->Visits;
              });

    std::cerr << "root value: " << root_->Value / root_->Visits << '\n'
              << "total simulations: " << numOfSimulations_ << '\n'
              << "root visits: " << root_->Visits << '\n';

    for (const TreeNode* child : children)
    {
        if (child->Visits == 0)
            continue;

        std::cerr << std::right << std::setw(5)
                  << Utils::ActionStr(child->Action)
                  << " : (N: " << child->Visits
                  << ") (Q: " << (child->Value / child->Visits)
                  << ") (P: " << child->Policy << ") -> ";

        while (child->State == ExpandState::EXPANDED)
        {
            child = child->GetMaxVisitedChild();

            std::cerr << Utils::ActionStr(child->Action) << ' ';
        }

        std::cerr << '\n';
    }
}

baba_is_auto::Direction SearchEngine::GetBestAction() const
{
    const TreeNode* bestNode = root_->GetMaxVisitedChild();

    return bestNode->Action;
}

const TreeNode* SearchEngine::GetRoot() const noexcept
{
    return root_;
}

std::shared_ptr<Train::TrainingSet> SearchEngine::CreateTrainingSet() const
{
    const int width = mainGame_->GetMap().GetWidth();
    const int height = mainGame_->GetMap().GetHeight();

    return std::make_shared<Train::TrainingSet>(width, height);
}

Train::TrainingData SearchEngine::GetTrainingData() const
{
    Train::TrainingData data;

    data.state = baba_is_auto::Preprocess::StateToTensor(*mainGame_);

    data.pi.resize(Utils::ACTION_SPACE.size());
    {
        float totalVisits = 1e-10f;
        root_->ForEach([&totalVisits](TreeNode* child) {
            totalVisits += child->Visits.load();
        });

        root_->ForEach([&data, totalVisits](TreeNode* child) {
            const int actionId = static_cast<int>(child->Action);

            data.pi[actionId] = child->Visits.load() / totalVisits;
        });
    }

    return data;
}

baba_is_auto::PlayState SearchEngine::GetResult() const
{
    return mainGame_->GetPlayState();
}

void SearchEngine::loadConfig()
{
    config_ = Common::Config(configFileName_);
}

void SearchEngine::updateRoot(TreeNode* newNode)
{
    TreeNode* node;
    if (newNode == nullptr)
    {
        node = new TreeNode;
    }
    else
    {
        node = new TreeNode(std::move(*newNode));

        node->ForEach([node](TreeNode* child) { child->ParentNode = node; });

        node->ParentNode = nullptr;
    }

    if (root_ != nullptr)
        deleteNode(root_);

    root_ = node;
}

void SearchEngine::initRoot()
{
    if (root_->State == ExpandState::UNEXPANDED)
    {
        Network::Tensor policy;
        [[maybe_unused]] float value;
        netMgr_.Evaluate(*mainGame_, policy, value);

        root_->Expand(policy);
    }

    if (config_.EnableDirichletNoise)
    {
        std::gamma_distribution<float> dist(config_.DirichletNoiseAlpha);

        const std::size_t numOfChildren = root_->NumOfChildren;
        std::vector<float> noise(numOfChildren);

        for (std::size_t i = 0; i < numOfChildren; ++i)
            noise[i] = effolkronium::random_thread_local::get(dist);

        const float noiseSum =
            std::accumulate(begin(noise), end(noise), 1e-10f);

        float total = 1e-10f;
        std::size_t idx = 0;
        root_->ForEach([&idx, &noise, &total, noiseSum, this](TreeNode* child) {
            child->Policy =
                (config_.DirichletNoiseEps) * child->Policy +
                (1 - config_.DirichletNoiseEps) * noise[idx] / noiseSum;

            total += child->Policy;
            ++idx;
        });

        root_->ForEach([total](TreeNode* child) { child->Policy /= total; });
    }
}

void SearchEngine::deleteNode(TreeNode* node)
{
    std::scoped_lock lock(deleteMutex_);

    deleteBuffer_.emplace_back(node);
}

void SearchEngine::pauseSearch()
{
    if (controller_.GetState() == SearchState::SEARCHING)
    {
        controller_.Pause();
    }
}

void SearchEngine::resumeSearch()
{
    if (controller_.GetState() == SearchState::PAUSE)
    {
        numOfSimulations_ = 0;
        initRoot();

        controller_.Resume();
    }
}

void SearchEngine::searchThread()
{
    if (!controller_.WaitResume())
        return;

    while (true)
    {
        if (controller_.GetState() != SearchState::SEARCHING)
        {
            controller_.AckPause();

            if (!controller_.WaitResume())
            {
                break;
            }
        }

        baba_is_auto::Game game(*mainGame_);

        TreeNode* tempNowNode = root_;
        while (tempNowNode->State == ExpandState::EXPANDED)
        {
            tempNowNode = tempNowNode->SelectChildNode(config_.cPUCT,
                                                       config_.InitPenalty);

            game.MovePlayer(tempNowNode->Action);
            Utils::AtomicAdd(tempNowNode->VirtualLoss, config_.VirtualLoss);
        }

        float valueToUpdate = 0;
        const auto gameState = game.GetPlayState();

        if (gameState != baba_is_auto::PlayState::WON && gameState != baba_is_auto::PlayState::LOST)
        {
            Network::Tensor policy;
            netMgr_.Evaluate(game, policy, valueToUpdate);

            tempNowNode->Expand(policy);
        }
        else
        {
            valueToUpdate = (gameState == baba_is_auto::PlayState::WON);
        }

        while (tempNowNode != nullptr)
        {
            Utils::AtomicAdd(tempNowNode->Value, valueToUpdate);
            ++tempNowNode->Visits;

            if (tempNowNode != root_)
            {
                Utils::AtomicAdd(tempNowNode->VirtualLoss,
                                 -config_.VirtualLoss);
            }

            tempNowNode = tempNowNode->ParentNode;
        }

        ++numOfSimulations_;
    }
}

void SearchEngine::deleteThread()
{
    const auto deleteImpl = [](TreeNode* node) {
        static void (*impl)(TreeNode*) = [](TreeNode* node) {
            node->ForEach([](TreeNode* child) { impl(child); });

            TreeNode* tempNowNode = node->MostLeftChildNode;
            TreeNode* nodeToDelete = nullptr;
            while (tempNowNode != nullptr)
            {
                nodeToDelete = tempNowNode;
                tempNowNode = nodeToDelete->RightSiblingNode;

                delete nodeToDelete;
            }

            node->MostLeftChildNode = nullptr;
        };

        impl(node);
    };

    while (isDeleteThreadRunning_)
    {
        std::unique_lock lock(deleteMutex_);

        if (deleteBuffer_.empty())
        {
            lock.unlock();

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        TreeNode* nodeToDelete = deleteBuffer_.front();
        deleteBuffer_.pop_front();

        deleteImpl(nodeToDelete);
        delete nodeToDelete;
    }
}
}  // namespace BabaIsAgent::Search
