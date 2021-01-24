#include <BabaIsAgent/Utils/Barrier.hpp>

#include <stdexcept>

namespace BabaIsAgent::Utils
{
void Barrier::Init(int count)
{
    std::scoped_lock lock(mutex_);

    count_ = count;
}

void Barrier::Done()
{
    {
        std::scoped_lock lock(mutex_);

        if (--count_ < 0)
            throw std::runtime_error("counter cannot be negative");
    }

    if (count_ == 0)
        cv_.notify_all();
}

void Barrier::Wait()
{
    std::unique_lock lock(mutex_);

    cv_.wait(lock, [&] { return count_ == 0; });
}
}  // namespace BabaIsAgent::Utils
