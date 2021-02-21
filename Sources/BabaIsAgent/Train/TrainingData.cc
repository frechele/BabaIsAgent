#include <BabaIsAgent/Train/TrainingData.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace BabaIsAgent::Train
{
TrainingSet::TrainingSet(int width, int height) : width_(width), height_(height)
{
}

void TrainingSet::Add(TrainingData data)
{
    data_.emplace_back(std::move(data));
}

void TrainingSet::SetResult(baba_is_auto::PlayState result)
{
    result_ = (result == baba_is_auto::PlayState::WON);
}

int TrainingSet::GetResult() const
{
    return result_;
}

void TrainingSet::Save(const std::string& filename) const
{
    std::ofstream fs(filename);
    if (!fs.is_open())
    {
        throw std::runtime_error("Cannot open file (" + filename + ")");
    }

	fs << width_ << ' ' << height_ << '\n';
	fs << result_ << '\n';

	for (const auto& data : data_)
	{
		for (float s : data.state)
		{
			fs << static_cast<int>(s) << ' ';
		}
		fs << '\n';

		for (float pi : data.pi)
		{
			fs << pi << ' ';
		}
		fs << '\n';
	}

    fs.close();
}
}  // namespace BabaIsAgent::Train
