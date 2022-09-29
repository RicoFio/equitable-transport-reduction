import igraph as ig
from .base_reward import BaseReward


class WelfareReward(BaseReward):

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def _reward_scaling(self, reward: float) -> float:
        return reward
