#ifndef BABA_IS_AGENT_SEARCH_ENGINE_HPP
#define BABA_IS_AGENT_SEARCH_ENGINE_HPP

#include <BabaIsAgent/Common/Config.hpp>
#include <BabaIsAgent/Network/NetworkManager.hpp>
#include <BabaIsAgent/Search/SearchController.hpp>
#include <BabaIsAgent/Search/TreeNode.hpp>
#include <BabaIsAgent/Train/TrainingData.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace BabaIsAgent::Search
{
class SearchEngine final
{
 public:
    //! Constructor with config file name.
    SearchEngine(std::string_view configFileName);

    //! Destructor.
    ~SearchEngine() noexcept;

    void Init(std::string_view gameFileName);

    void DoSearch();
    void Play(baba_is_auto::Direction action);

    void DumpStats() const;

    [[nodiscard]] baba_is_auto::Direction GetBestAction() const;
    [[nodiscard]] const TreeNode* GetRoot() const noexcept;

    // For Training
    std::shared_ptr<Train::TrainingSet> CreateTrainingSet() const;
    Train::TrainingData GetTrainingData() const;
    baba_is_auto::PlayState GetResult() const;

 private:
    void loadConfig();

    void updateRoot(TreeNode* newNode);
    void initRoot();

    void deleteNode(TreeNode* node);

    void pauseSearch();
    void resumeSearch();

    void searchThread();
    void deleteThread();

 private:
    std::string_view configFileName_;
    Common::Config config_;

    std::unique_ptr<baba_is_auto::Game> mainGame_{ nullptr };
    TreeNode* root_{ nullptr };

    Network::NetworkManager netMgr_;

    // Search Thread
    SearchController controller_;
    std::vector<std::thread> searchWorkers_;

    // Delete Thread
    bool isDeleteThreadRunning_{ true };
    std::deque<TreeNode*> deleteBuffer_;
    std::mutex deleteMutex_;
    std::thread deleteWorker_;

    // Statistics
    std::atomic<std::size_t> numOfSimulations_{ 0 };
};
}  // namespace BabaIsAgent::Search

#endif  // BABA_IS_AGENT_SEARCH_ENGINE_HPP
