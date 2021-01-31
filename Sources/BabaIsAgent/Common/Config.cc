#include <BabaIsAgent/Common/Config.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

#define CONFIG(type, name) \
	name = j.value<type>(#name, name)

namespace BabaIsAgent::Common
{
Config::Config(std::string_view configFileName)
{
	std::ifstream file(configFileName.data());
	if (!file.is_open())
		throw std::runtime_error("cannot open config file");

	nlohmann::json j;
	file >> j;

	// Network Options
	CONFIG(std::string, WeightFileName);

	CONFIG(int, NumOfEvalWorker);
	CONFIG(int, BatchSize);
	CONFIG(std::vector<int>, Gpus);

	// Search Options
	CONFIG(int, NumOfSearchWorker);

	CONFIG(float, DirichletNoiseAlpha);
	CONFIG(float, DirichletNoiseEps);
	CONFIG(bool, EnableDirichletNoise);
	CONFIG(std::size_t, MaxSimulationCount);

	CONFIG(float, cPUCT);
	CONFIG(float, InitPenalty);
	CONFIG(float, VirtualLoss);
}
}  // namespace BabaIsAgent::Common

#undef CONFIG
