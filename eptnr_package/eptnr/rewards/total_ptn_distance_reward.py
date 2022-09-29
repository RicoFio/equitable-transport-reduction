import igraph as ig
from .base_reward import BaseReward


class TotalPTNDistanceReward(BaseReward):

    def _reward_scaling(self, reward: float) -> float:
        return reward

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        """

        Args:
            g:
            *args:
            **kwargs:

        Returns:

        """
        return sum(g.es.select(type_in=['train','bus'],active_eq=1)['distance'])
