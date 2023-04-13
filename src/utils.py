import json
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def save_stats(stats):
    with open("stats.json", "w") as f:
        json.dump(stats, f, sort_keys=True, indent=4)


def load_stats() -> Dict:
    with open("stats.json", "r") as f:
        return json.load(f)


def print_stats(stats):
    print(json.dumps(stats, indent=4, sort_keys=True))


def highlight_nodes(graph: nx.Graph, nodes: List = []):
    reset_colors(graph)
    for node in nodes:
        graph.nodes[node]["viz"]["color"] = {"r": 255, "g": 0, "b": 0, "a": 1}


def highlight_nodes_importance(graph: nx.Graph, node_importance: Dict = {}):
    reset_colors(graph)
    min = np.min(list(node_importance.values()))
    max = np.max(list(node_importance.values()))
    norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
    for node, importance in node_importance.items():
        g = int(255 * norm(importance))
        r = 255 - g
        graph.nodes[node]["viz"]["color"] = {"r": r, "g": g, "b": 0, "a": 1}


def highlight_nodes_communities(graph: nx.Graph, communities: List[List] = []):
    reset_colors(graph)
    cmap: LinearSegmentedColormap = plt.get_cmap("hsv")
    for i, community in enumerate(communities):
        (r, g, b, a) = cmap(i / len(communities))
        r, g, b, a = int(r * 255), int(g * 255), int(b * 255), a
        for node in community:
            graph.nodes[node]["viz"]["color"] = {"r": r, "g": g, "b": b, "a": a}


def reset_colors(graph: nx.Graph):
    for node in graph:
        if not "viz" in graph.nodes[node]:
            graph.nodes[node]["viz"] = {}
        graph.nodes[node]["viz"]["color"] = {"r": 173, "g": 216, "b": 230, "a": 1}


def circular_layout(graph: nx.Graph):
    pos = nx.layout.circular_layout(graph, scale=4 * graph.number_of_nodes())
    for node, position in pos.items():
        if not "viz" in graph.nodes[node]:
            graph.nodes[node]["viz"] = {}
        graph.nodes[node]["viz"]["position"] = {
            "x": position[0],
            "y": position[1],
            "z": 0.0,
        }
