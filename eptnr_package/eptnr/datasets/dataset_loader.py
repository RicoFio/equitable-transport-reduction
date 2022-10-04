from pathlib import Path
from enum import Enum
import igraph as ig
import geopandas as gdp
from typing import Tuple, Union
import pickle


class SyntheticDatasets(Enum):
    ONE = Path('./synthetic_1')
    TWO = Path('./synthetic_2')
    THREE = Path('./synthetic_3')
    FOUR = Path('./synthetic_4')
    FIVE = Path('./synthetic_5')


def load_dataset(which: SyntheticDatasets) -> Tuple[ig.Graph, Union[gdp.GeoDataFrame, dict]]:
    """
    Returns graph and [GeoDataFrame OR RewardDict] depending on the dataset:
        - ONE: Graph & GDF
        - TWO: Graph & GDF
        - THREE: Graph & GDF
        - FOUR: Graph & RewardDict
        - FIVE: Graph & RewardDict
    """
    graph = ig.read(which.value / 'graph.picklez')

    if which in [SyntheticDatasets.ONE, SyntheticDatasets.TWO, SyntheticDatasets.THREE]:
        return graph, gdp.read_file(which.value / 'census_data.geojson', engine='GeoJSON')
    else:
        return graph, pickle.load(which.value / 'reward_dict.pkl')
