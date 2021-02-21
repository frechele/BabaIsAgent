#include <BabaIsAgent/Search/SearchEngine.hpp>

#include <atomic>
#include <csignal>
#include <ctime>
#include <iostream>
#include <sstream>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

#include <lyra/lyra.hpp>

namespace
{
std::atomic<std::size_t> numOfDoneGames{ 0 };
bool isRunning = false;
void sigHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
        isRunning = false;
}

std::string makeDateStr()
{
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);

    std::stringstream ss;

    ss << (now->tm_year + 1900) << '_' << (now->tm_mon + 1) << '_'
       << now->tm_mday << '/' << now->tm_hour << '_' << now->tm_min << '_'
       << now->tm_sec;

    return ss.str();
}

void SelfThread(int threadId, const std::string& configFileName)
{
    using namespace BabaIsAgent;

    while (isRunning)
    {
        Search::SearchEngine engine(configFileName.data());
        engine.Init("game.txt");

        auto dataset = engine.CreateTrainingSet();

        auto result = baba_is_auto::PlayState::INVALID;
        while (true)
        {
            if (result == baba_is_auto::PlayState::LOST ||
                result == baba_is_auto::PlayState::WON)
                break;

            engine.DoSearch();
            dataset->Add(engine.GetTrainingData());

            engine.Play(engine.GetBestAction());
        }
        dataset->SetResult(result);

        const std::string dataDir = "traindata/" + makeDateStr() + "_" + std::to_string(threadId) + ".dat";
        dataset->Save(dataDir);

        ++numOfDoneGames;
    }
}
}  // namespace

namespace BabaIsAgent::Command
{
int RunSelf(int argc, char** argv)
{
    bool showHelp = false;
    std::string configFileName;
    std::string mapFileName;
    std::size_t numOfGameThread;

    auto cli =
        lyra::cli() | lyra::help(showHelp) |
        lyra::opt(configFileName,
                  "config")["--config"]("Configuration file path")
            .required() |
        lyra::opt(mapFileName, "map")["--map"]("Map file path").required() |
        lyra::opt(numOfGameThread,
                  "threads")["--thread"]("The number of self game threads")
            .required();

    auto result = cli.parse({ argc, argv });
    if (!result)
    {
        std::cerr << result.errorMessage() << std::endl;
        std::cerr << cli << std::endl;

        return EXIT_FAILURE;
    }

    if (showHelp)
    {
        std::cerr << cli << std::endl;

        return EXIT_SUCCESS;
    }

    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    if (!fs::exists("traindata"))
        fs::create_directory("traindata");

    isRunning = true;

    std::vector<std::thread> gameWorkers(numOfGameThread);
    for (int threadId = 0; threadId < static_cast<int>(numOfGameThread);
         ++threadId)
    {
        gameWorkers[threadId] =
            std::thread(SelfThread, threadId, std::ref(configFileName));
    }

    while (isRunning)
    {
        const std::size_t numGames = numOfDoneGames.load();

        if (numGames % 10 == 9)
        {
            std::cout << numGames << " games done." << std::endl;
        }
    }

    for (auto& worker : gameWorkers)
    {
        if (worker.joinable())
            worker.join();
    }

    return EXIT_SUCCESS;
}
}  // namespace BabaIsAgent::Command
