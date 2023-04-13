from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple, TypedDict

import networkx as nx
import numpy as np
import pandas as pd


class Stats(TypedDict):
    real_len: int
    calculated_len: int
    intersection_len: int
    accuracy: float


class Score(TypedDict):
    sum_real: int
    sum_intersection: int
    accuracy: float


def load_features(src: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    df = df[["language", "numeric_id"]]
    return df


def remove_features(df: pd.DataFrame, features_to_remove: List[str]) -> pd.DataFrame:
    df = df[~df["language"].isin(features_to_remove)]
    df.reset_index(inplace=True, drop=True)
    return df


def keep_nodes(df: pd.DataFrame, number_of_nodes_to_keep: int) -> pd.DataFrame:
    indexes_to_keep = np.random.choice(df.index, size=number_of_nodes_to_keep)
    df = df.iloc[indexes_to_keep]
    df.reset_index(inplace=True, drop=True)
    return df


def load_edges(src: str, features: pd.DataFrame) -> pd.DataFrame:
    edges = pd.read_csv(src)

    ids = features["numeric_id"]

    edges_1 = edges[edges["id_1"].isin(ids)]
    edges_2 = edges[edges["id_2"].isin(ids)]
    index = pd.Index.intersection(edges_1.index, edges_2.index)
    edges = edges.iloc[index]
    edges.reset_index(inplace=True, drop=True)
    return edges


def create_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in edges.iterrows():
        G.add_edge(row[1]["id_1"], row[1]["id_2"])
    return G


def choose_largest_cc(graph: nx.Graph) -> nx.Graph:
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)
    return graph


def get_real_communities(
    grah: nx.Graph, features: pd.DataFrame
) -> List[Tuple[str, Set[int]]]:
    real_communities: Dict[str, Set[int]] = defaultdict(set)
    for node in grah.nodes():
        language = features[features["numeric_id"] == node].iloc[0]["language"]
        real_communities[language].add(node)
    real_communities_list = [(k, v) for k, v in real_communities.items()]
    return sorted(real_communities_list, key=lambda x: len(x[1]), reverse=True)


def set_weights(
    graph: nx.Graph,
    metric: Dict[int, float],
    aggregate: Callable[[List[float]], float],
):
    weights = {}
    for edge in graph.edges():
        n1, n2 = edge
        m1, m2 = metric[n1], metric[n2]
        weights[edge] = aggregate([m1, m2])

    nx.set_edge_attributes(graph, weights, "weights")


def calculate_communities(graph: nx.Graph) -> List[Set[int]]:
    calculated_communities = list(
        nx.algorithms.community.asyn_lpa_communities(graph, weight="weights")
    )
    calculated_communities = sorted(calculated_communities, key=len, reverse=True)
    return calculated_communities


def calculate_mapping_stats(
    real_communities: List[Tuple[str, Set[int]]], calculated_communities: List[Set[int]]
) -> Dict[str, Stats]:
    result: Dict[str, Stats] = {}

    already_mapped = set()
    for i in range(min(len(real_communities), len(calculated_communities))):
        calculated = calculated_communities[i]

        best = {
            "language": "",
            "real_len": -1,
            "intersection_len": -1,
            "accuracy": -1,
        }

        for language, real in real_communities:
            if language in already_mapped:
                continue

            intersection_len = len(real.intersection(calculated))
            accuracy = intersection_len / (
                len(real) + len(calculated) - intersection_len
            )

            if accuracy > best["accuracy"]:
                best = {
                    "language": language,
                    "real_len": len(real),
                    "intersection_len": intersection_len,
                    "accuracy": accuracy,
                }

        result[best["language"]] = {
            "real_len": best["real_len"],
            "calculated_len": len(calculated),
            "intersection_len": best["intersection_len"],
            "accuracy": best["accuracy"],
        }
        already_mapped.add(best["language"])

    return result


def calculate_score(mapping_stats: Dict[str, Stats]) -> Score:
    sum_real = 0
    sum_intersection = 0
    sum_calculated = 0
    for stats in mapping_stats.values():
        sum_real += stats["real_len"]
        sum_intersection += stats["intersection_len"]
        sum_calculated += stats["calculated_len"]
    return {
        "sum_real": sum_real,
        "sum_intersection": sum_intersection,
        "sum_calculated": sum_calculated,
        "accuracy": sum_intersection / max(sum_real, sum_calculated),
    }


def evaluate_metric(
    G: nx.Graph, real_communities: List[Tuple[str, Set[int]]], iterations: int
):
    scores = []
    for _ in range(iterations):
        calculated_communities: List[Set[int]] = calculate_communities(G)
        mapping_stats = calculate_mapping_stats(
            real_communities, calculated_communities
        )
        score = calculate_score(mapping_stats)["accuracy"]
        scores.append(score)
    return scores
