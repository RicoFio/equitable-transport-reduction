import igraph as ig
from ._utils import series_min_max_norm
from ..constants.travel_metric import TravelMetric
from .base_reward import BaseReward
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class UtilitarianReward(BaseReward):

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        """

        Args:
            com_threshold:
            g:
            census_data:
            groups:

        Returns:

        """
        metrics_dfs = self.retrieve_dfs(g)

        tt_samples_hat = metrics_dfs[TravelMetric.TT.value]
        hops_samples_hat = metrics_dfs[TravelMetric.HOPS.value]
        com_samples_hat = metrics_dfs[TravelMetric.COM.value]

        # Normalization
        if kwargs.get('normalization', False):
            tt_samples_hat['metric_value'] = series_min_max_norm(tt_samples_hat.metric_value.astype(float))
            hops_samples_hat['metric_value'] = series_min_max_norm(hops_samples_hat.metric_value.astype(float))
            com_samples_hat['metric_value'] = series_min_max_norm(com_samples_hat.metric_value.astype(float))

        reward = 0

        for group in self.groups:
            tt_reward = float(tt_samples_hat[tt_samples_hat.group == group]['metric_value'].values.astype(float).mean())
            tt_reward *= TravelMetric.TT.value in self.metrics_names
            hops_reward = float(hops_samples_hat[hops_samples_hat.group == group]['metric_value'].values.astype(float).mean())
            hops_reward *= TravelMetric.HOPS.value in self.metrics_names
            com_reward = float(com_samples_hat[com_samples_hat.group == group]['metric_value'].values.astype(float).mean())
            com_reward *= TravelMetric.COM.value in self.metrics_names

            group_reward = - tt_reward - hops_reward + com_reward
            # logger.info(f"computed reward {group_reward} for group {group}")
            reward += group_reward

        return reward

    def _reward_scaling(self, reward: float) -> float:
        return reward
