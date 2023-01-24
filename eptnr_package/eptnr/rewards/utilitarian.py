import igraph as ig
import numpy as np
import pandas as pd
from typing import List
from .utils.chebyshev_reward_computation import PartialRewardGenerator, chebyshev_reward_computation

from .utils.graph_computation_utils import series_min_max_norm
from ..constants.travel_metric import TravelMetric
from .base_reward import BaseReward
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class UtilitarianReward(BaseReward):

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        def _obtain_reward(metric: TravelMetric) -> np.ndarray:
            samples_hat = metrics_dfs[metric.value]
            if kwargs.get('normalization', False):
                samples_hat['metric_value'] = series_min_max_norm(samples_hat.metric_value.astype(float))

            r = []
            for group in self.groups:
                r.append(float(
                    samples_hat[samples_hat.group == group]['metric_value'].values.astype(float).mean()
                ))
            return np.array(r)

        rewards = np.zeros((len(self.metrics_names), len(self.groups)))

        for i, metric in enumerate(self.metrics_names):
            reward = _obtain_reward(TravelMetric[metric.upper()])
            if TravelMetric.COM.value != metric:
                reward = -reward
            rewards[i, :] = reward

        final_reward = np.concatenate(rewards).sum()

        return final_reward

    def _reward_scaling(self, reward: float) -> float:
        return reward


class UtilitarianCostReward(BaseReward):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float, max_travel_time: float,
                 total_graph_cost: float, monetary_budget: float,
                 groups: List[str] = None, metrics: List[TravelMetric] = None,
                 verbose: bool = False):
        super().__init__(census_data, com_threshold, groups, metrics, verbose)
        self.monetary_budget = monetary_budget
        self.utilitarian_reward = UtilitarianReward(census_data, com_threshold, groups, metrics, verbose, reward_scaling=False)
        self.utilitarian_rg = PartialRewardGenerator(0, max_travel_time)
        if total_graph_cost - monetary_budget <= 0:
            raise ValueError("The monetary budget cannot be bigger than the cost of the total graph")
        self.cost_rg = PartialRewardGenerator(0, total_graph_cost)

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def _reward_scaling(self, reward: float) -> float:
        return reward

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        utilitarian_reward = self.utilitarian_reward.evaluate(g, *args, **kwargs)

        total_savings = sum(g.es.select(active=0)['cost'])

        partial_utilitarian_reward = self.utilitarian_rg.generate_reward(utilitarian_reward)
        partial_cost_reward = self.cost_rg.generate_reward(abs(total_savings - self.monetary_budget))

        final_reward = chebyshev_reward_computation(partial_utilitarian_reward, partial_cost_reward)

        return final_reward
