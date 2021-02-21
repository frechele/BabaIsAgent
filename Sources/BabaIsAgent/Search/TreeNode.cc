#include <BabaIsAgent/Search/TreeNode.hpp>

#include <BabaIsAgent/Utils/Utils.hpp>

#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>

namespace BabaIsAgent::Search
{
TreeNode::TreeNode(TreeNode&& other) noexcept
    : State(other.State.load()),
      Action(other.Action),
      Policy(other.Policy),
      Visits(other.Visits.load()),
      Value(other.Value.load()),
      VirtualLoss(other.VirtualLoss.load()),
      NumOfChildren(other.NumOfChildren),
      ParentNode(other.ParentNode),
      MostLeftChildNode(other.MostLeftChildNode)
{
    other.State = ExpandState::UNEXPANDED;
    other.MostLeftChildNode = nullptr;
    other.NumOfChildren = 0;
}

TreeNode* TreeNode::SelectChildNode(float cPUCT, float initPenalty)
{
    // Wait for expanding
    while (State.load() == ExpandState::EXPANDING)
        ;

    assert(MostLeftChildNode != nullptr);

    float totalParentVisits = 0;
    ForEach([&totalParentVisits](TreeNode* child) {
        totalParentVisits += child->Visits.load();
    });

    float sqrtTotalParentVisits = std::sqrt(totalParentVisits);

    float maxValue = -FLT_MAX;
    TreeNode* maxChild = nullptr;

    ForEach([&maxValue, &maxChild, sqrtTotalParentVisits, cPUCT,
             initPenalty](TreeNode* child) {
        const int visits = child->Visits.load();
        const float policy = child->Policy;
        const float value = child->Value.load();
        const float virtualloss = child->VirtualLoss.load();

        const float Qvalue =
            (value - initPenalty) / (visits + initPenalty + virtualloss);
        const float Uvalue =
            cPUCT * policy * sqrtTotalParentVisits / (1.f + visits);

        const float score = Qvalue + Uvalue;

        if (maxValue < score)
        {
            maxValue = score;
            maxChild = child;
        }
    });

    assert(maxChild != nullptr);
    return maxChild;
}  // namespace BabaIsAgent::Search

void TreeNode::Expand(const Network::Tensor& policy)
{
    {
        ExpandState expected = ExpandState::UNEXPANDED;
        if (!State.compare_exchange_strong(expected, ExpandState::EXPANDING))
            return;
    }

    const std::size_t numOfActions = Utils::ACTION_SPACE.size();
    TreeNode* nowNode = nullptr;
    for (std::size_t i = 0; i < numOfActions; ++i)
    {
        TreeNode* node = new TreeNode;
        node->Action = Utils::ACTION_SPACE[i];
        node->Policy = policy[i];

        if (nowNode == nullptr)
            MostLeftChildNode = node;
        else
            nowNode->RightSiblingNode = node;

        ++NumOfChildren;
        node->ParentNode = this;
        nowNode = node;
    }

    State = ExpandState::EXPANDED;
}

TreeNode* TreeNode::GetMaxVisitedChild()
{
    return const_cast<TreeNode*>(std::as_const(*this).GetMaxVisitedChild());
}

const TreeNode* TreeNode::GetMaxVisitedChild() const
{
    int maxVisits = -INT_MAX;
    const TreeNode* maxChild = nullptr;

    ForEach([&maxVisits, &maxChild](const TreeNode* child) {
        const int visits = child->Visits.load();

        if (maxVisits < visits)
        {
            maxVisits = visits;
            maxChild = child;
        }
    });

    assert(maxChild != nullptr);
    return maxChild;
}
}  // namespace BabaIsAgent::Search
