"""
This script demonstrates how to compute the complexity and tonicity scores of triads.
"""

from math import exp, sqrt
from statistics import harmonic_mean
from tkinter import N


NOTE_NAMES = ["C", "E", "G"]

TONICITY_CONTEXT = [1/3]*3
"""
The pre-existing tonicity context for how "tonic" each of the three notes are in the triad.
"""

DYADIC_COMP = [
    0.759, # C-E
    0.764, # E-G
    0.512, # C-G
]
"""
Complexity between the the 1st-2nd, 2nd-3rd, and 1st-3rd pairs of notes in triad.
"""

TREES = [
    # With 1st note as root
    [
        {
            0: [1, 2],
        },
        {
            0: [1],
            1: [2],
        },
        {
            0: [2],
            2: [1]
        }
    ],
    # With 2nd note as root
    [
        {
            1: [0, 2],
        },
        {
            1: [0],
            0: [2],
        },
        {
            1: [2],
            2: [0]
        }
    ],
    # With 3rd note as root
    [
        {
            2: [0, 1],
        },
        {
            2: [0],
            0: [1],
        },
        {
            2: [1],
            1: [0]
        }
    ],
]
"""
Hardcoded list of all minimal spanning trees for triads.

Each key-value pair are (from, to) edges in the direction from root to leaves.
"""

def get_dyadic_complexity(from_idx, to_idx):
    """
    Get dyadic complexity between two note indices.
    """
    if (from_idx == 0 and to_idx == 1) or (from_idx == 1 and to_idx == 0):
        return DYADIC_COMP[0]
    elif (from_idx == 1 and to_idx == 2) or (from_idx == 2 and to_idx == 1):
        return DYADIC_COMP[1]
    elif (from_idx == 0 and to_idx == 2) or (from_idx == 2 and to_idx == 0):
        return DYADIC_COMP[2]
    else:
        raise ValueError("Invalid note indices")

def dfs(idx, tree, ctx=TONICITY_CONTEXT):
    """
    DFS to compute complexity and tonicity

    The base case is a leaf node which returns TONICITY_CONTEXT[idx] tonicity and None subtree
    complexity.

    Returns (tonicity, complexity) of the subtree rooted at idx, where tonicity is the sum of
    per-note tonicities in the subtree in [0, 1], and subtree complexity is in [0,1].
    """

    children = tree.get(idx, [])
    if not children:
        return ctx[idx], None

    child_tonicities = []
    child_complexities = []

    for child in children:
        child_tonicity, child_comp = dfs(child, tree, ctx)
        child_tonicities.append(child_tonicity)
        edge_comp = get_dyadic_complexity(idx, child)
        if child_comp is None:
            unweighted_comp = edge_comp
        else:
            unweighted_comp = (edge_comp + child_comp) * 0.5 * (1 - 0.5 * (edge_comp - child_comp))
        child_complexities.append(unweighted_comp)

        # print(f"{NOTE_NAMES[idx]} -> {NOTE_NAMES[child]}: edge_comp={edge_comp:.5f}, child_comp={child_comp}, unweighted_comp={unweighted_comp:.5f}")

    total_tonicity = sum(child_tonicities)
    normalized_child_tonicities = [ct / total_tonicity for ct in child_tonicities]
    softmax_sum = sum(exp(t) for t in normalized_child_tonicities)
    softmax_tonicities = [exp(t) / softmax_sum for t in normalized_child_tonicities]

    total_complexity = sum(child_complexities[i] * softmax_tonicities[i] for i in range(len(children)))

    return total_tonicity, total_complexity

def print_tree(tree: dict, root):
    """
    Print tree in terminal.

    Tree is given as a dict of (from, to) edges.
    """

    def print_recursive(idx, depth=0):
        if depth == 0:
            print(NOTE_NAMES[idx])
        else:
            print("    " * (depth-1) + "|---" + NOTE_NAMES[idx])
        children = tree.get(idx, [])
        for i, child in enumerate(children):
            print_recursive(child, depth + 1)
    print_recursive(root)


def test_one_iteration():
    """
    Perform one iteration of complexity computation for all triad MSTs.

    This method tries out different ways of aggregating complexity scores per choice of root.

    Does not update the TONICITY_CONTEXT.

    For a plain C-E-G triad, it is expected that choosing E as root gives significantly higher
    complexity.

    However, between C and G, these are both closely matched with C only having slightly lower
    complexity.

    I'm interested in which of the different aggregation methods (arithmetic & harmonic &
    exp-weighted means) gives the most separation between C and G.

    Arithmetic & harmonic means are simply the AM and HM of complexities.

    IEM is the inverse exponentially-weighted mean where lower complexities are given exponentially
    more weight.

    EM is the exponenentially-weighted mean where higher complexities are given exponentially more
    weight.

    I compare the aggregated scores w.r.t. C as the baseline.

    For root C:
    - AM: 0.6509
    - HM: 0.64056
    - IEM: 0.64407
    - EM: 0.65789

    For root E:
    - AM: 0.73129 (diff: -0.08039)
    - HM: 0.73067 (diff: -0.09011)
    - IEM: 0.73083 (diff: -0.08676)
    - EM: 0.73175 (diff: -0.07386)

    For root G:
    - AM: 0.65281 (diff: -0.00191)
    - HM: 0.64204 (diff: -0.00148)
    - IEM: 0.64569 (diff: -0.00162)
    - EM: 0.66008 (diff: -0.00219)

    Notice the tradeoff between the separation between C and G and between C and E. I decide to go
    with the exponentially weighted mean where higher complexities have more weight.
    """
    comp_per_root = [0.0] * len(TREES)
    for root in range(3):
        complexity_scores = []
        for tree in TREES[root]:
            print_tree(tree, root)
            _, complexity = dfs(root, tree)
            if complexity is None:
                print("Complexity is None (leaf node only)")
                continue
            print(f"Complexity: {complexity:.5f}")

            complexity_scores.append(complexity)
            print("___________________________")

        arith_mean = sum(complexity_scores) / len(complexity_scores)
        print(f"Arithmetic mean complexity for root {NOTE_NAMES[root]}: {arith_mean:.5f}")

        harmonic_mean = len(TREES[root]) / sum(1/c for c in complexity_scores)
        print(f"Harmonic mean complexity for root {NOTE_NAMES[root]}: {harmonic_mean:.5f}")

        total_inv_exp_weight = sum(exp(1-c) for c in complexity_scores)
        inv_exp_weighted_mean = sum(c * exp(1-c) / total_inv_exp_weight for c in complexity_scores)
        print(f"Inverse exp. weighted mean complexity for root {NOTE_NAMES[root]}: {inv_exp_weighted_mean:.5f}")

        total_exp_weight = sum(exp(c) for c in complexity_scores)
        exp_weighted_mean = sum(c * exp(c) / total_exp_weight for c in complexity_scores)
        print(f"Exp. weighted mean complexity for root {NOTE_NAMES[root]}: {exp_weighted_mean:.5f}")
        print("\n\n")


def test_tonicity_update(iterations: int = 30, smoothing: float = 0.7, temperature: float = 0.5):
    """
    Test tonicity update over multiple iterations. The computed complexities are used to compute
    target tonicities, and these are fed back to the complexity computation.

    `temperature` controls how opinionated the tonicity scores are. Higher temperature = less opinionated.

    The target tonicity scores are computed as the softmax of (1 - agg_comp[root]) where
    agg_comp[root] is the aggregated complexity score for all interpretation trees with root `root`.

    By the experiment in `test_one_iteration`, I decide to use the exp. weighted mean as the
    aggregation method.

    This test ensures that even with a feedback loop where tonicity context is passed back into the
    complexity computation, the tonicity scores stabilizes.
    """

    ctx = TONICITY_CONTEXT.copy()

    for it in range(iterations):
        agg_comp_per_root = [0.0] * len(TREES)
        for root in range(3):
            complexity_scores = []
            for tree in TREES[root]:
                _, complexity = dfs(root, tree, ctx)
                if complexity is None:
                    continue
                complexity_scores.append(complexity)

            total_exp_weight = sum(exp(c) for c in complexity_scores)
            exp_weighted_mean = sum(c * exp(c) / total_exp_weight for c in complexity_scores)
            agg_comp_per_root[root] = exp_weighted_mean

        target_tonicities = [exp((1 - agg_comp_per_root[r]) / temperature) for r in range(3)]
        softmax_sum = sum(target_tonicities)
        target_tonicities = [tt / softmax_sum for tt in target_tonicities]

        # Smooth update
        for i in range(3):
            ctx[i] = smoothing * ctx[i] + (1 - smoothing) * target_tonicities[i]

        # Normalize ctx
        total_ctx = sum(ctx)
        ctx = [c / total_ctx for c in ctx]

        print(f"Iteration {it+1:<6} target: {[f'{tt:.5f}' for tt in target_tonicities]} ctx: {[f'{c:.5f}' for c in ctx]}")


# test_tonicity_update()

test_one_iteration()