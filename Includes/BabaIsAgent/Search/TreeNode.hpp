#ifndef BABA_IS_AGENT_TREE_NODE_HPP
#define BABA_IS_AGENT_TREE_NODE_HPP

#include <baba-is-auto/baba-is-auto.hpp>

#include <BabaIsAgent/Network/Network.hpp>

#include <atomic>
#include <memory>

namespace BabaIsAgent::Search
{
enum class ExpandState
{
    UNEXPANDED,
    EXPANDING,
    EXPANDED
};

struct TreeNode final
{
    //! Default constructor.
    TreeNode() = default;

    //! Move constructor.
    TreeNode(TreeNode&& other) noexcept;

    //! Move assignment operator.
    TreeNode& operator=(TreeNode&& other);

    //! Delete copy constructor.
    TreeNode(const TreeNode&) = delete;

    //! Delete copy assignment operator.
    TreeNode& operator=(const TreeNode&) = delete;

    std::atomic<ExpandState> State{ ExpandState::UNEXPANDED };

    baba_is_auto::Direction Action{ baba_is_auto::Direction::NONE };

    float Policy{ 0 };
    std::atomic<int> Visits{ 0 };
    std::atomic<float> Value{ 0 };
    std::atomic<float> VirtualLoss{ 0 };

    std::size_t NumOfChildren{ 0 };
    TreeNode* ParentNode{ nullptr };
    TreeNode* MostLeftChildNode{ nullptr };
    TreeNode* RightSiblingNode{ nullptr };

    TreeNode* SelectChildNode(float cPUCT, float initPenalty);
    void Expand(const Network::Tensor& policy);

    TreeNode* GetMaxVisitedChild();

    template <typename Func>
    void ForEach(Func&& func);

    template <typename Func>
    void ForEach(Func&& func) const;
};

template <typename Func>
void TreeNode::ForEach(Func&& func)
{
    TreeNode* tempNowNode = MostLeftChildNode;

    while (tempNowNode != nullptr)
    {
        func(tempNowNode);

        tempNowNode = tempNowNode->RightSiblingNode;
    }
}

template <typename Func>
void TreeNode::ForEach(Func&& func) const
{
    const TreeNode* tempNowNode = MostLeftChildNode;

    while (tempNowNode != nullptr)
    {
        func(tempNowNode);

        tempNowNode = tempNowNode->RightSiblingNode;
    }
}
}  // namespace BabaIsAgent::Search

#endif  // BABA_IS_AGENT_TREE_NODE_HPP
