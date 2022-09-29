import igraph as ig
from .utilitarian import UtilitarianReward
from .base_reward import BaseReward
import pandas as pd
from typing import List
from ..constants.travel_metric import TravelMetric


class ElitarianReward(BaseReward):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float,
                 groups: List[str] = None, metrics: List[TravelMetric] = None,
                 verbose: bool = False):
        if len(groups) != 1:
            raise ValueError(f"The elite can be only one group, not {groups or 'empty list'}")
        super().__init__(census_data, com_threshold, groups, metrics, verbose)

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        ur = UtilitarianReward(census_data=self.census_data, groups=self.groups,
                               com_threshold=self.com_threshold)
        return ur.evaluate(g)

    def _reward_scaling(self, reward: float) -> float:
        return reward
