from typing import List, Tuple
import random
import igraph as ig
from ...rewards import BaseReward


def random_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                    budget: int = 5) -> Tuple[List[float], List[int]]:
    """

    Args:
        reward:
        edge_types:
        g:
        budget:

    Returns:

    """
    g_prime = g.copy()

    assert 0 < budget < len(g.es.select(type_in=edge_types))

    removed_edges = []
    rewards_per_removal = []

    for i in range(budget):
        removable_edges = g_prime.es.select(type_in=edge_types, active_eq=1)
        edge_to_remove = random.sample(list(removable_edges), 1)[0].index
        removed_edges.append(edge_to_remove)

        g_prime.es[edge_to_remove]['active'] = 0
        r = reward.evaluate(g_prime)
        rewards_per_removal.append(r)

    return rewards_per_removal, removed_edges
