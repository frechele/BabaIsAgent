#include <BabaIsAgent/Common/Config.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

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
	WeightFileName = j.value<std::string>("WeightFileName", WeightFileName);

	NumOfEvalWorker = j.value<int>("NumOfEvalWorker", NumOfEvalWorker);
	BatchSize = j.value<int>("BatchSize", BatchSize);
	Gpus = j.value<std::vector<int>>("Gpus", Gpus);

	// Search Options
	NumOfSearchWorker = j.value<int>("NumOfSearchWorker", NumOfSearchWorker);

	cPUCT = j.value<float>("cPUCT", cPUCT);
	InitPenalty = j.value<float>("InitPenalty", InitPenalty);
	VirtualLoss = j.value<float>("VirtualLoss", VirtualLoss);
}
}  // namespace BabaIsAgent::Common
