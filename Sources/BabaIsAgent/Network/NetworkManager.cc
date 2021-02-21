#include <BabaIsAgent/Network/NetworkManager.hpp>

#include <BabaIsAgent/Network/RandomNetwork.hpp>
#include <BabaIsAgent/Network/TrtNetwork.hpp>

namespace BabaIsAgent::Network
{
NetworkManager::NetworkManager(const Common::Config& config)
    : config_(config), workers_(config.NumOfEvalWorker)
{
}

NetworkManager::~NetworkManager()
{
    isRunning_ = false;

    cv_.notify_all();

    for (auto& worker : workers_)
    {
        if (worker.joinable())
            worker.join();
    }
}

void NetworkManager::Init()
{
    const int numOfWorkers = config_.NumOfEvalWorker;
    [[maybe_unused]] const std::size_t numOfGpus = config_.Gpus.size();

    barrier_.Init(numOfWorkers);

    isRunning_ = true;

    for (int threadId = 0; threadId < numOfWorkers; ++threadId)
    {
        std::unique_ptr<Network> network;

#ifdef USE_TENSORRT
        network =
            std::make_unique<TrtNetwork>(config_.Gpus[threadId % numOfWorkers]);
#else
        network = std::make_unique<RandomNetwork>();
#endif

        workers_[threadId] =
            std::thread(&NetworkManager::evalThread, this, std::move(network));
    }

    barrier_.Wait();
}

void NetworkManager::Evaluate(const baba_is_auto::Game& state, Tensor& policy,
                              float& value)
{
    Task task;
    task.input = baba_is_auto::Preprocess::StateToTensor(state);
    auto fut = task.output.get_future();

    {
        std::scoped_lock lock(mutex_);

        buffer_.emplace_back(std::move(task));
        ++size_;
    }

    cv_.notify_all();

    NetResult result = fut.get();

    policy = std::move(result.policy);
    value = result.value;
}

void NetworkManager::evalThread(std::unique_ptr<Network> network)
{
    network->Initialize(config_.WeightFileName);
    barrier_.Done();

    while (isRunning_)
    {
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return !isRunning_ || size_.load() > 0; });
        }

        std::vector<Tensor> inputs;
        std::vector<std::promise<NetResult>> outputs;

        std::size_t batchSize;

        {
            std::scoped_lock lock(mutex_);

            batchSize =
                std::min<std::size_t>(config_.BatchSize, buffer_.size());

            if (batchSize == 0)
                continue;

            for (std::size_t batch = 0; batch < batchSize; ++batch)
            {
                Task task = std::move(buffer_.front());
                buffer_.pop_front();

                inputs.emplace_back(std::move(task.input));
                outputs.emplace_back(std::move(task.output));
            }
            size_ -= batchSize;
        }

        std::vector<Tensor> policy(batchSize);
        Tensor value(batchSize);

        network->Evaluate(inputs, policy, value);

        for (std::size_t batch = 0; batch < batchSize; ++batch)
        {
            outputs[batch].set_value(
                { std::move(policy[batch]), value[batch] });
        }
    }
}
}  // namespace BabaIsAgent::Network
