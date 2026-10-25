"""Compute graph answers in code so the model can never grade against a wrong key."""
from __future__ import annotations

from typing import Any

import networkx as nx


def build_graph(block: dict) -> nx.Graph:
    g = nx.Graph()
    for n in block.get("nodes", []):
        nid = n["id"] if isinstance(n, dict) else n
        g.add_node(nid)
    for e in block.get("edges", []):
        g.add_edge(e["u"], e["v"], weight=float(e["w"]))
    return g


def shortest_path(block: dict, source: str, target: str) -> dict[str, Any]:
    g = build_graph(block)
    path = nx.shortest_path(g, source, target, weight="weight")
    cost = nx.shortest_path_length(g, source, target, weight="weight")
    next_hop = path[1] if len(path) > 1 else source
    return {"path": path, "cost": cost, "next_hop": next_hop}


def forwarding_table(block: dict, source: str) -> dict[str, dict[str, Any]]:
    g = build_graph(block)
    table: dict[str, dict[str, Any]] = {}
    for target in sorted(g.nodes):
        if target == source:
            continue
        try:
            info = shortest_path(block, source, target)
            table[target] = {"next_hop": info["next_hop"], "cost": info["cost"], "path": info["path"]}
        except nx.NetworkXNoPath:
            table[target] = {"next_hop": None, "cost": None, "path": []}
    return table


def answers_for_lesson(lesson: dict) -> dict[str, str]:
    """Resolve every blank/question answer. Explicit answers win; solvers fill the rest."""
    answers: dict[str, str] = {}
    graphs = {b["id"]: b for b in lesson.get("blocks", []) if b.get("type") == "graph"}

    for block in lesson.get("blocks", []):
        btype = block.get("type")
        if btype == "text":
            for run in block.get("runs", []):
                if "blank" in run and "answer" in run:
                    answers[run["blank"]] = str(run["answer"])
        elif btype == "graph":
            for e in block.get("edges", []):
                if "blank" in e:
                    answers[e["blank"]] = str(e["w"])
            for n in block.get("nodes", []):
                if isinstance(n, dict) and "blank" in n:
                    answers[n["blank"]] = str(n.get("label", n["id"]))
        elif btype == "question":
            bid = block.get("blank") or block["id"]
            if "answer" in block:
                answers[bid] = str(block["answer"])
                continue
            solver = block.get("solver")
            graph = graphs.get(block.get("graph", ""))
            if not graph or not solver:
                continue
            source = block.get("source", graph.get("source"))
            target = block.get("target")
            if solver == "next_hop" and source and target:
                answers[bid] = shortest_path(graph, source, target)["next_hop"]
            elif solver == "cost" and source and target:
                answers[bid] = str(int(shortest_path(graph, source, target)["cost"]))
            elif solver == "path" and source and target:
                answers[bid] = " → ".join(shortest_path(graph, source, target)["path"])
        elif btype == "table":
            graph = graphs.get(block.get("graph", ""))
            source = block.get("source")
            if not graph or not source:
                continue
            table = forwarding_table(graph, source)
            for row in block.get("rows", []):
                dest = row["dest"]
                info = table.get(dest, {})
                if "blank_hop" in row and info.get("next_hop") is not None:
                    answers[row["blank_hop"]] = str(info["next_hop"])
                if "blank_cost" in row and info.get("cost") is not None:
                    answers[row["blank_cost"]] = str(int(info["cost"]))
        elif btype in ("flow", "tree", "compare", "conceptmap"):
            for item in block.get("blanks", []):
                if "id" in item and "answer" in item:
                    answers[item["id"]] = str(item["answer"])
    return answers
