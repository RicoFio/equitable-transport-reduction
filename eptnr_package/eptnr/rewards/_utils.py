import igraph as ig
import pandas as pd
from typing import Tuple
import numpy as np
import multiprocessing
import ray
from typing import List

CPU_COUNT = multiprocessing.cpu_count()


class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


@ray.remote
class GraphActor(Singleton):

    instance = None

    def __init__(self, graph: ig.Graph):
        if not self.instance:
            self.graph = graph

    def get_instance(self):
        return self

    def set_graph(self, v):
        self.graph = v

    def get_graph(self):
        return self.graph


@ray.remote
def get_shortest_path(graph_actor: GraphActor, v_rc_ids: List[int], v_poi_ids: List[int]):
    graph = ray.get(graph_actor.get_graph.remote())
    v_rcs = graph.vs[v_rc_ids]
    v_poi_list = graph.vs[v_poi_ids]
    return graph.shortest_paths(v_rcs, v_poi_list, weights='tt')


@ray.remote
def get_shortest_path_length(graph_actor: GraphActor, v_rcs: List[ig.Vertex],
                             v_poi_list: ig.EdgeSeq) -> List[List[int]]:
    graph = ray.get(graph_actor.get_graph.remote())
    shortest_paths = []
    for v_rc in v_rcs:
        shortest_path_edges = graph.get_shortest_paths(v_rc, v_poi_list, weights='tt', output='epath')
        shortest_paths.append([len(es) for es in shortest_path_edges])

    return shortest_paths


def _to_array(ray_list: List[ray.ObjectRef], n_nb_nodes: int, n_poi_nodes: int) -> np.array:
    batch_computation = np.vstack(
        [np.array(ray.get(rows)).flatten().reshape(len(ray.get(rows)), n_poi_nodes) for rows in ray_list]
    )
    assert batch_computation.shape == (n_nb_nodes, n_poi_nodes)
    return np.vstack(batch_computation)


def evaluate_graph(g: ig.Graph) -> pd.DataFrame:
    nb_nodes = g.vs.select(type_eq='rc_node')
    # nb_node_ids = [elem.index for elem in nb_nodes]
    # n_nb_nodes = len(nb_nodes)
    poi_nodes = g.vs.select(type_eq='poi_node')
    # poi_node_ids = [elem.index for elem in poi_nodes]
    # n_poi_nodes = len(poi_nodes)

    shortest_paths_tt = g.shortest_paths(nb_nodes, poi_nodes, weights='tt')

    # Travel Time
    tt_mx = np.array(shortest_paths_tt)
    # Assign max over both dimensions to inf values
    tt_mx[tt_mx == np.inf] = tt_mx.max(1).max()

    # Number of hops
    hops_mx = []
    for v_rc in nb_nodes:
        shortest_path_edges = g.get_shortest_paths(v_rc, poi_nodes, weights='tt', output='epath')
        hops_mx.append([len(es) for es in shortest_path_edges])

    # hops_mx_ray = []
    #
    # BATCH_SIZE = 100
    # global_graph_actor = GraphActor.remote(graph=g)
    # for i in range(0, len(nb_node_ids), BATCH_SIZE):
    #     batch_nb_node_ids = nb_node_ids[i: i + BATCH_SIZE]
    #
    #     shortest_path_edges = get_shortest_path_length.remote(global_graph_actor, batch_nb_node_ids, poi_node_ids)
    #     hops_mx_ray.append(shortest_path_edges)
    #
    # hops_mx = _to_array(hops_mx_ray, n_nb_nodes, n_poi_nodes)

    hops_mx = np.array(hops_mx)
    hops_mx[hops_mx == 0] = 1

    poi_node_names = [f'POI_{name}' if 'POI' not in name else name for name in poi_nodes['name']]
    rc_node_names = [f'RC_{name}' if 'RC' not in name else name for name in nb_nodes['name']]

    df_tt = pd.DataFrame(tt_mx, columns=poi_node_names)
    df_tt['metric'] = 'travel_time'
    df_tt['rc'] = rc_node_names

    df_hops = pd.DataFrame(hops_mx, columns=poi_node_names)
    df_hops['metric'] = 'hops'
    df_hops['rc'] = rc_node_names

    return pd.concat([df_tt, df_hops], axis=0)


def generate_samples(metric_df: pd.DataFrame, inh_per_group: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(metric_df, inh_per_group, how='left', left_on='rc', right_on='neighborhood')

    city = pd.DataFrame(columns=['metric_value', 'group'])

    for group in merged_df.loc[:, merged_df.columns.str.contains('n_')].columns:
        samples_group = np.repeat(merged_df['average'].values, merged_df[group])
        samples_group = samples_group.reshape((-1, 1))
        labels = np.repeat(group.replace('n_', ''), samples_group.shape[0]).reshape((-1, 1))
        group_individuals = np.hstack([samples_group, labels])
        df = pd.DataFrame(group_individuals, columns=['metric_value', 'group'])
        city = pd.concat([city, df], axis=0)

    return city


def get_tt_hops_com_dfs(g: ig.Graph, census_data: pd.DataFrame,
                        com_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_df = evaluate_graph(g)
    groups_census = census_data.drop(columns=['n_inh', 'geometry'])

    metric_df['average'] = metric_df.loc[:, metric_df.columns.str.contains('POI')].mean(axis=1)
    tt_samples = generate_samples(metric_df[metric_df['metric'] == 'travel_time'], groups_census)
    hops_samples = generate_samples(metric_df[metric_df['metric'] == 'hops'], groups_census)

    value_cols = metric_df.loc[:, metric_df.columns.str.contains('POI')]
    metric_df['average'] = (value_cols < com_threshold).sum(axis=1)
    com_samples = generate_samples(metric_df[metric_df['metric'] == 'travel_time'], groups_census)

    return tt_samples, hops_samples, com_samples


def series_min_max_norm(sr: pd.Series):
    return (sr - sr.min()) / (sr.max() - sr.min())
