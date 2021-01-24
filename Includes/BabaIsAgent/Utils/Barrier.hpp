#ifndef BABA_IS_AGENT_BARRIER_HPP
#define BABA_IS_AGENT_BARRIER_HPP

#include <condition_variable>
#include <mutex>

namespace BabaIsAgent::Utils
{
class Barrier final
{
 public:
	void Init(int size);

	void Done();

	void Wait();

 private:
	std::mutex mutex_;
    std::condition_variable cv_;

	int count_;
};
}  // namespace BabaIsAgent::Utils

#endif  // BABA_IS_AGENT_BARRIER_HPP
