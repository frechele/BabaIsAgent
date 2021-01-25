#ifndef BABA_IS_AGENT_CONFIG_HPP
#define BABA_IS_AGENT_CONFIG_HPP

#include <string>
#include <string_view>
#include <vector>

namespace BabaIsAgent::Common
{
struct Config final
{
    //! Default cosntructor.
    Config() = default;

    //! Load from config file.
    Config(std::string_view configFileName);

    // Network Options
    std::string WeightFileName;

    int NumOfEvalWorker{ 1 };
    int BatchSize{ 1 };
    std::vector<int> Gpus;

    // Search Options
    int NumOfSearchWorker{ 1 };

    bool EnableDirichletNoise{ false };
    std::size_t MaxSimulationCount{ 10 };

    float cPUCT{ 1.8f };
    float InitPenalty{ 0 };
    float VirtualLoss{ 0 };
};
}  // namespace BabaIsAgent::Common

#endif  // BABA_IS_AGENT_CONFIG_HPP
