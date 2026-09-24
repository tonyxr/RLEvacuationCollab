#!/usr/bin/env python3
"""Exact reusable graph primitives for evacuation routing optimization."""

from __future__ import annotations

import networkx as nx


def distances_to_target(graph, target, *, weight: str = "length") -> dict:
    """Compute every directed origin-to-target distance with one search.

    Reversing a directed graph maps each original origin-to-target path to a
    target-to-origin path of identical weight. The input graph is exposed
    through a read-only reverse view and is never mutated.
    """
    if target not in graph:
        return {}
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    return dict(
        nx.single_source_dijkstra_path_length(
            reverse_graph,
            target,
            weight=weight,
        )
    )


def routing_tree_to_target(graph, target, *, weight: str = "length") -> tuple[dict, dict]:
    """Return exact distances and one deterministic next hop toward ``target``.

    A single Dijkstra search on the reverse graph replaces one shortest-path
    search per pedestrian. In the reverse predecessor relation, each origin's
    predecessor is precisely its next node on an original-graph path to the
    target. Equal-distance alternatives use stable node ordering.
    """
    if target not in graph:
        return {}, {}
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    predecessors, distances = nx.dijkstra_predecessor_and_distance(
        reverse_graph,
        target,
        weight=weight,
    )

    def stable_node_key(node):
        try:
            return (0, int(node))
        except (TypeError, ValueError):
            return (1, str(node))

    # Convert the shortest-path predecessor DAG into a rooted routing tree.
    # Growing outward from the target guarantees that following ``next_hop``
    # always terminates even if a source graph contains zero-length ties.
    children = {}
    for origin, options in predecessors.items():
        for parent in options:
            children.setdefault(parent, []).append(origin)
    next_hop = {}
    discovered = {target}
    frontier = [target]
    cursor = 0
    while cursor < len(frontier):
        parent = frontier[cursor]
        cursor += 1
        for child in sorted(children.get(parent, ()), key=stable_node_key):
            if child in discovered:
                continue
            discovered.add(child)
            next_hop[child] = parent
            frontier.append(child)
    return dict(distances), next_hop
