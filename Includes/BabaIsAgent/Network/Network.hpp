#ifndef BABA_IS_AGENT_NETWORK_HPP
#define BABA_IS_AGENT_NETWORK_HPP

#include <string>
#include <vector>

namespace BabaIsAgent::Network
{
using Tensor = std::vector<float>;

struct NetResult final
{
    Tensor policy;
    float value;
};

class Network
{
 public:
    //! Default destructor.
    virtual ~Network() = default;

    virtual void Initialize(const std::string& weightFileName) = 0;

    virtual void Evaluate(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& policy, Tensor& value) = 0;
};
}  // namespace BabaIsAgent::Network

#endif  // BABA_IS_AGENT_NETWORK_HPP
