#ifndef BABA_IS_AGENT_TRT_NETWORK_HPP
#define BABA_IS_AGENT_TRT_NETWORK_HPP

#ifdef USE_TENSORRT

#include <BabaIsAgent/Common/Config.hpp>
#include <BabaIsAgent/Network/Network.hpp>

#include <NvInfer.h>

namespace BabaIsAgent::Network
{
class TrtNetwork final : public Network
{
 public:
    TrtNetwork(int gpuId);
    ~TrtNetwork();

    void Initialize(const std::string& weightFileName) override;

    void Evaluate(const std::vector<Tensor>& inputs,
                  std::vector<Tensor>& policy, Tensor& value) override;

 private:
	nvinfer1::IRuntime* runtime_{ nullptr };
	nvinfer1::ICudaEngine* engine_{ nullptr };
	nvinfer1::IExecutionContext* context_{ nullptr };

	std::vector<void*> buffers_;

	int gpuId_;
};
}  // namespace BabaIsAgent::Network

#endif  // USE_TENSORRT

#endif  // BABA_IS_AGENT_TRT_NETWORK_HPP
