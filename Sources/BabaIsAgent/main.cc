#include <BabaIsAgent/Command/Commands.hpp>

#include <iostream>
#include <string_view>

int main(int argc, char** argv)
{
    using namespace BabaIsAgent;

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <command> <options>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    const std::string_view command{ argv[1] };

    const int cmdArgc = argc - 1;
    char** const cmdArgv = argv + 1;

    if (command == "self")
        return Command::RunSelf(cmdArgc, cmdArgv);

    // Invalid command
    std::cerr << "ERROR: invalid command (" << command << ")" << std::endl;

    return EXIT_FAILURE;
}
