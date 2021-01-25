#ifndef BABA_IS_AGENT_WAIT_GROUP_HPP
#define BABA_IS_AGENT_WAIT_GROUP_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace BabaIsAgent::Utils
{
class WaitGroup final
{
 public:
    void Add(int incr = 1);
    void Done();
    void Wait();

 private:
    std::mutex mutex_;
    std::atomic<int> counter_{ 0 };
    std::condition_variable cv_;
};
}  // namespace BabaIsAgent::Utils

#endif  // BABA_IS_AGENT_WAIT_GROUP_HPP