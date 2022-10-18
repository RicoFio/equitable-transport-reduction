import igraph as ig
import numpy as np
import pandas as pd
from typing import List, Dict
from .utils.chebyshev_reward_computation import PartialRewardGenerator, chebyshev_reward_computation

from ..constants.travel_metric import TravelMetric
from .base_reward import BaseReward
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SufficientarianReward(BaseReward):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float, sufficiency: Dict[TravelMetric, float],
                 groups: List[str] = None, metrics: List[TravelMetric] = None, verbose: bool = False):
        super().__init__(census_data, com_threshold, groups, metrics, verbose)
        self.total_population = self.census_data.n_inh.sum()
        self.sufficiency = sufficiency

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        def _obtain_reward(metric: TravelMetric) -> np.ndarray:
            samples_hat = metrics_dfs[metric.value]

            metric_values = samples_hat['metric_value'].values.astype(float)
            if TravelMetric.COM == metric:
                metric_counts = metric_values[metric_values > self.sufficiency[metric]].shape
            else:
                metric_counts = metric_values[metric_values < self.sufficiency[metric]].shape
            r = self.total_population - metric_counts
            return np.array(r)

        rewards = np.zeros((len(self.metrics_names), 1))

        for i, metric in enumerate(self.metrics_names):
            reward = _obtain_reward(TravelMetric[metric.upper()])
            rewards[i, :] = reward

        final_reward = -np.concatenate(rewards).sum()

        return final_reward

    def _reward_scaling(self, reward: float) -> float:
        return reward


class SufficientarianCostReward(BaseReward):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float, sufficiency: Dict[TravelMetric, float],
                 total_graph_cost: float, monetary_budget: float,
                 groups: List[str] = None, metrics: List[TravelMetric] = None,
                 verbose: bool = False):
        super().__init__(census_data, com_threshold, groups, metrics, verbose)
        self.monetary_budget = monetary_budget
        self.sufficientarian_reward = SufficientarianReward(census_data, com_threshold, sufficiency,
                                                            groups, metrics, verbose)
        total_inhabitants = self.census_data.n_inh.sum()
        self.sufficientarian_rg = PartialRewardGenerator(0, total_inhabitants)
        if total_graph_cost - monetary_budget <= 0:
            raise ValueError("The monetary budget cannot be bigger than the cost of the total graph")
        self.cost_rg = PartialRewardGenerator(0, total_graph_cost)

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def _reward_scaling(self, reward: float) -> float:
        return reward

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        sufficientarian_reward = -self.sufficientarian_reward.evaluate(g, *args, **kwargs)

        total_savings = sum(g.es.select(active=0)['cost'])

        partial_sufficientarian_reward = self.sufficientarian_rg.generate_reward(sufficientarian_reward)
        partial_cost_reward = self.cost_rg.generate_reward(abs(total_savings - self.monetary_budget))

        final_reward = -chebyshev_reward_computation(partial_sufficientarian_reward, partial_cost_reward)

        return final_reward
