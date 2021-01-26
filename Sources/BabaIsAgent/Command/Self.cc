#include <iostream>

#include <lyra/lyra.hpp>

namespace BabaIsAgent::Command
{
int RunSelf(int argc, char** argv)
{
    bool showHelp = false;
    std::string configFileName;

    auto cli = lyra::cli() | lyra::help(showHelp) |
               lyra::opt(configFileName,
                         "config")["--config"]("Configuration file path")
                   .required();

    auto result = cli.parse({ argc, argv });
	if (!result)
	{
		std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
		std::cerr << cli << std::endl;

		return EXIT_FAILURE;
	}

	if (showHelp)
	{
		std::cerr << cli << std::endl;

		return EXIT_SUCCESS;
	}

    return EXIT_SUCCESS;
}
}  // namespace BabaIsAgent::Command
