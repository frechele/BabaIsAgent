#ifdef USE_TENSORRT

#include <BabaIsAgent/Network/TrtNetwork.hpp>

#include <BabaIsAgent/Utils/Utils.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std::string_literals;

namespace
{
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                throw std::runtime_error(msg);
            default:
                std::cerr << "[TensorRT] "s << msg << std::endl;
        }
    }
} TRT_LOGGER;
}  // namespace

namespace BabaIsAgent::Network
{
TrtNetwork::TrtNetwork(int gpuId) : gpuId_(gpuId)
{
}

TrtNetwork::~TrtNetwork()
{
    if (context_)
        context_->destroy();
    if (engine_)
        engine_->destroy();
    if (runtime_)
        runtime_->destroy();

    for (auto buffer : buffers_)
    {
        if (int ret = cudaFree(buffer); ret != 0)
        {
            std::cerr << "CUDA free error (code: " << ret << ")" << std::endl;
        }
    }
}

void TrtNetwork::Initialize(const std::string& weights)
{
    cudaSetDevice(gpuId_);

    std::ostringstream oss(std::ios::binary);
    if (!(oss << std::ifstream(weights, std::ios::binary).rdbuf()))
    {
        throw std::runtime_error("Cannot open weights");
    }
    std::string model = oss.str();

    runtime_ = nvinfer1::createInferRuntime(TRT_LOGGER);
    engine_ =
        runtime_->deserializeCudaEngine(model.c_str(), model.size(), nullptr);
    if (engine_ == nullptr)
    {
        throw std::runtime_error("Cannot create engine");
    }

    context_ = engine_->createExecutionContext();

    const int batchSize = engine_->getMaxBatchSize();
    for (int i = 0; i < engine_->getNbBindings(); ++i)
    {
        const auto dim = engine_->getBindingDimensions(i);

        int size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }

        void* buffer;
        if (int ret = cudaMalloc(&buffer, batchSize * size * sizeof(float)))
        {
            throw std::runtime_error("CUDA out of memory (code: "s +
                                     std::to_string(ret) + ")"s);
        }

        buffers_.emplace_back(buffer);
    }
}

void TrtNetwork::Evaluate(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& policy, Tensor& value)
{
	const std::size_t batchSize = inputs.size();
	assert(batchSize > 0);

	const std::size_t TENSOR_SIZE = inputs[0].size();

	Tensor inTensor(batchSize * TENSOR_SIZE);
	for (std::size_t batch = 0; batch < batchSize; ++batch)
	{
		std::copy(begin(inputs[batch]), end(inputs[batch]), begin(inTensor) + TENSOR_SIZE * batch);
	}

	if (int ret =
            cudaMemcpy(buffers_[0], inTensor.data(),
                       inTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
        ret != 0)
    {
        throw std::runtime_error("CUDA memcpy error (code: "s +
                                 std::to_string(ret) + ")"s);
    }

	context_->executeV2(buffers_.data());

	const std::size_t actionSize = Utils::ACTION_SPACE.size();
	Tensor flatPolicy(batchSize * actionSize);
	if (int ret = cudaMemcpy(flatPolicy.data(), buffers_[1],
                             flatPolicy.size() * sizeof(float),
                             cudaMemcpyDeviceToHost);
        ret != 0)
    {
        throw std::runtime_error("CUDA memcpy error (code: "s +
                                 std::to_string(ret) + ")"s);
    }

	for (std::size_t batch = 0; batch < batchSize; ++batch)
	{
		policy[batch].resize(actionSize);

		for (std::size_t i = 0; i < actionSize; ++i)
		{
			policy[batch][i] = flatPolicy[batch * actionSize + i];
		}
	}

	if (int ret =
            cudaMemcpy(value.data(), buffers_[2], value.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        ret != 0)
    {
        throw std::runtime_error("CUDA memcpy error (code: "s +
                                 std::to_string(ret) + ")"s);
    }
}
}  // namespace BabaIsAgent::Network

#endif  // USE_TENSORRT
