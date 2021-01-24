#ifndef BABA_IS_AGENT_UTILS_HPP
#define BABA_IS_AGENT_UTILS_HPP

#include <baba-is-auto/baba-is-auto.hpp>

#include <array>
#include <algorithm>
#include <atomic>

namespace BabaIsAgent::Utils
{
template <typename T>
void AtomicAdd(std::atomic<T>& t, T value)
{
    T oldValue = t.load();

    while (!t.compare_exchange_weak(oldValue, oldValue + value))
        ;
}

template <typename ContainerT>
std::size_t Argmax(const ContainerT& container)
{
    return std::distance(
        std::begin(container),
        std::max_element(std::begin(container), std::end(container)));
}

template <typename ContainerT, typename CompareFunc>
std::size_t Argmax(const ContainerT& container, CompareFunc&& func)
{
    return std::distance(
        std::begin(container),
        std::max_element(std::begin(container), std::end(container), func));
}

inline constexpr std::array<baba_is_auto::Direction, 5> ACTION_SPACE = {
    baba_is_auto::Direction::NONE,
    baba_is_auto::Direction::UP,
    baba_is_auto::Direction::DOWN,
    baba_is_auto::Direction::LEFT,
    baba_is_auto::Direction::RIGHT
};
}  // namespace BabaIsAgent::Utils

#endif  // BABA_IS_AGENT_UTILS_HPP