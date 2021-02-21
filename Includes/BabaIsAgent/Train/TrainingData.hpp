#ifndef BABA_IS_AGENT_TRAINING_DATA_HPP
#define BABA_IS_AGENT_TRAINING_DATA_HPP

#include <BabaIsAgent/Network/Network.hpp>

#include <baba-is-auto/baba-is-auto.hpp>

#include <string>
#include <vector>

namespace BabaIsAgent::Train
{
struct TrainingData final
{
    Network::Tensor state;
    Network::Tensor pi;
};

class TrainingSet final
{
 public:
    TrainingSet(int width, int height);

    void Add(TrainingData data);

    void SetResult(baba_is_auto::PlayState result);
    int GetResult() const;

    void Save(const std::string& filename) const;

 private:
    int width_, height_;
    int result_{ 0 };
    std::vector<TrainingData> data_;
};
}  // namespace BabaIsAgent::Train

#endif  // BABA_IS_AGENT_TRAINING_DATA_HPP
