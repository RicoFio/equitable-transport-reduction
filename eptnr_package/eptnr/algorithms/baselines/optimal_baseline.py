import itertools as it
from typing import Tuple, List
import igraph as ig
import numpy as np
from ...rewards import BaseReward
from tqdm import tqdm
import logging
from .utils.compute_rewards import compute_rewards_over_removals

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _optimal_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str]) -> List[Tuple[List[float], List[int]]]:
    """

    Args:
        g:
        reward:
        edge_types:

    Returns:

    """
    assert 0 < len(g.es.select(type_in=edge_types))

    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    possible_combinations = [[e.index for e in es]
                             for k in range(len(removable_edges))
                             for es in it.combinations(removable_edges, k)]

    # logger.info(f"Possible states: {possible_combinations}")
    rewards = -np.ones(len(possible_combinations)) * np.inf

    for i, candidate in enumerate(tqdm(possible_combinations)):
        g_prime = g.copy()
        g_prime.es[candidate]['active'] = 0
        rewards[i] = reward.evaluate(g_prime)
        logger.info(f"For state {candidate} obtained rewards {rewards[i]}")

    max_reward_candidates_idxs = np.where(rewards == rewards.max())[0]

    optimal_solutions_and_rewards_per_removal = []
    logger.info("OPTIMAL STATES:")
    for cand_i in max_reward_candidates_idxs:
        logger.info(f"For state {possible_combinations[cand_i]} obtained rewards {rewards[cand_i]}")
        es_idx_list = possible_combinations[cand_i]
        rewards_per_removal = compute_rewards_over_removals(g, reward, es_idx_list)
        optimal_solutions_and_rewards_per_removal.append((rewards_per_removal, es_idx_list))

    return optimal_solutions_and_rewards_per_removal


def optimal_max_baseline(g: ig.Graph, reward: BaseReward,
                         edge_types: List[str]) -> List[Tuple[List[float], List[int]]]:
    """
    Args:
        g:
        reward:
        edge_types:

    Returns:
        List of optimal configuration reaching maximum rewards over all solutions in S as a
        list of rewards over each removal in that solution and the edges removed.
    """
    available_edges_to_remove = g.es.select(type_in=edge_types)
    assert 0 < len(available_edges_to_remove)

    all_opt = []
    for k in range(1, len(available_edges_to_remove)):
        opt_sol_rew_tuple_list = _optimal_baseline(g, reward, edge_types)
        all_opt.extend(opt_sol_rew_tuple_list)

    all_opt = np.array(all_opt, dtype=object)
    opt_idxs = np.argmax(all_opt[:, 0][-1]).tolist()
    opt_idxs = [opt_idxs] if isinstance(opt_idxs, int) else opt_idxs

    output = []

    for idx in opt_idxs:
        output.append(tuple(all_opt[idx, :].tolist()))

    return output
