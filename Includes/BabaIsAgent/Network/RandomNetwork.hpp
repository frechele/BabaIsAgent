#ifndef BABA_IS_AGENT_RANDOM_NETWORK_HPP
#define BABA_IS_AGENT_RANDOM_NETWORK_HPP

#include <BabaIsAgent/Network/Network.hpp>

namespace BabaIsAgent::Network
{
class RandomNetwork : public Network
{
 public:
    void Initialize(const std::string& weightFileName) override;

    void Evaluate(const std::vector<Tensor>& inputs,
                  std::vector<Tensor>& policy, Tensor& value) override;
};
}  // namespace BabaIsAgent::Network

#endif  // BABA_IS_AGENT_RANDOM_NETWORK_HPP
