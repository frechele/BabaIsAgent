#include <BabaIsAgent/Search/SearchController.hpp>

namespace BabaIsAgent::Search
{
SearchController::SearchController(std::size_t threadNum) : threadNum_(threadNum)
{
    // Do nothing
}

void SearchController::Pause()
{
    resumeGroup_.Wait();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        state_ = SearchState::PAUSE;
    }
    pauseGroup_.Wait();
}

void SearchController::AckPause()
{
    pauseGroup_.Done();
}

void SearchController::Resume()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == SearchState::SEARCHING)
            return;

        resumeGroup_.Add(threadNum_);
        pauseGroup_.Add(threadNum_);

        state_ = SearchState::SEARCHING;
    }

    cv_.notify_all();
    resumeGroup_.Wait();
}

void SearchController::Terminate()
{
    Pause();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        state_ = SearchState::TERMINATE;
    }

    cv_.notify_all();
}

bool SearchController::WaitResume()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] { return state_ != SearchState::PAUSE; });

        if (state_ == SearchState::TERMINATE)
            return false;
    }
    
    resumeGroup_.Done();
    return true;
}

SearchState SearchController::GetState() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return state_;
}
}
