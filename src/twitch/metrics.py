from typing import Dict

import networkx as nx


def metric_unitary(graph: nx.Graph) -> Dict[int, float]:
    metric = {}
    for node in graph.nodes():
        metric[node] = 1
    return metric


def metric_degree(graph: nx.Graph) -> Dict[int, float]:
    metric = {}
    for node, degree in graph.degree():
        metric[node] = degree
    return metric


def metric_degree_centrality(graph: nx.Graph) -> Dict[int, float]:
    return nx.degree_centrality(graph)


def metric_eigenvector_centrality(graph: nx.Graph) -> Dict[int, float]:
    return nx.eigenvector_centrality(graph)


def metric_pagerank(graph: nx.Graph) -> Dict[int, float]:
    return nx.pagerank(graph)


def metric_clustering(graph: nx.Graph) -> Dict[int, float]:
    return nx.clustering(graph)


def metric_closeness_centrality(graph: nx.Graph) -> Dict[int, float]:
    return nx.closeness_centrality(graph)


def metric_betweeness_centrality(graph: nx.Graph) -> Dict[int, float]:
    return nx.betweenness_centrality(graph)
