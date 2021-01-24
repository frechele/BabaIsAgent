#ifndef BABA_IS_AGENT_NETWORK_MANAGER_HPP
#define BABA_IS_AGENT_NETWORK_MANAGER_HPP

#include <BabaIsAgent/Common/Config.hpp>
#include <BabaIsAgent/Network/Network.hpp>
#include <BabaIsAgent/Utils/Barrier.hpp>

#include <baba-is-auto/baba-is-auto.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace BabaIsAgent::Network
{
class NetworkManager final
{
 public:
    //! Constructor with network.
    NetworkManager(const Common::Config& config);

    //! Destructor
    ~NetworkManager();

    //! Delete copy constructor.
    NetworkManager(const NetworkManager&) = delete;

    //! Delete move constructor.
    NetworkManager(NetworkManager&&) = delete;

    //! Delete copy assignment operator.
    NetworkManager& operator=(const NetworkManager&) = delete;

    //! Delete move assignment operator.
    NetworkManager& operator=(NetworkManager&&) = delete;

	void Init();

    void Evaluate(const baba_is_auto::Game& state, Tensor& policy,
                  float& value);

 private:
    struct Task final
    {
        Tensor input;
        std::promise<NetResult> output;
    };

 private:
    void evalThread(std::unique_ptr<Network> network);

 private:
	const Common::Config& config_;

    std::atomic<bool> isRunning_{ false };
    std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<std::size_t> size_{ 0 };
    std::deque<Task> buffer_;

	Utils::Barrier barrier_;
    std::vector<std::thread> workers_;
};
}  // namespace BabaIsAgent::Network

#endif  // BABA_IS_AGENT_NETWORK_MANAGER_HPP
