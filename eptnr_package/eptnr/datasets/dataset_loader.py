import igraph as ig
import geopandas as gdp
from typing import Tuple, Union
import pickle
from .synthetic_datasets import SyntheticDatasets


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
