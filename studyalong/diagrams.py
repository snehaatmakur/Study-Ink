"""Render structured diagrams from JSON. Structure from data; sketchy strokes; handwriting labels."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from handwriting import Writer

from .sketch import paste_centered, sketch_circle, sketch_line, sketch_rect


Blank = dict[str, Any]


def _label(
    page: Image.Image,
    writer: Writer,
    text: str,
    cx: float,
    cy: float,
    size: float,
    ink: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    frag, base, _ = writer.fragment(text, size=size)
    return paste_centered(page, frag, base, cx, cy, ink)


def render_graph(
    page: Image.Image,
    block: dict,
    writer: Writer,
    origin: tuple[int, int],
    ink: tuple[int, int, int],
    size: float,
    rng: random.Random,
    answers: dict[str, str],
    reveal: set[str] | None = None,
) -> tuple[list[Blank], tuple[int, int]]:
    """Draw a graph with fixed node coordinates. Returns blanks and (width, height) of the block."""
    reveal = reveal or set()
    ox, oy = origin
    draw = ImageDraw.Draw(page)
    nodes = {n["id"]: n for n in block["nodes"]}
    blanks: list[Blank] = []

    # edges first
    for e in block.get("edges", []):
        a, b = nodes[e["u"]], nodes[e["v"]]
        x0, y0 = ox + a["x"], oy + a["y"]
        x1, y1 = ox + b["x"], oy + b["y"]
        sketch_line(draw, (x0, y0, x1, y1), ink, width=2, rng=rng, roughness=1.2)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # offset label perpendicular to the edge
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5 or 1
        nx, ny = -dy / length * 16, dx / length * 16
        lx, ly = mx + nx, my + ny
        weight = str(e["w"])
        if "blank" in e and e["blank"] not in reveal:
            bw, bh = int(size * 1.1), int(size * 0.85)
            box = (int(lx - bw / 2), int(ly - bh / 2), int(lx + bw / 2), int(ly + bh / 2))
            sketch_rect(draw, box, ink, width=2, rng=rng)
            blanks.append(
                {
                    "id": e["blank"],
                    "kind": "edge_weight",
                    "bbox": list(box),
                    "answer": answers.get(e["blank"], weight),
                    "hint": f"Edge {e['u']}–{e['v']}",
                }
            )
        else:
            _label(page, writer, weight, lx, ly, size * 0.72, ink)

    # nodes
    r = block.get("node_radius", 28)
    for n in block["nodes"]:
        cx, cy = ox + n["x"], oy + n["y"]
        sketch_circle(draw, cx, cy, r, fill=(252, 250, 245), outline=ink, width=2, rng=rng)
        label = n.get("label", n["id"])
        if "blank" in n and n["blank"] not in reveal:
            bw, bh = int(r * 1.6), int(r * 1.1)
            box = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
            blanks.append(
                {
                    "id": n["blank"],
                    "kind": "node_label",
                    "bbox": list(box),
                    "answer": answers.get(n["blank"], label),
                    "hint": "Node label",
                }
            )
        else:
            _label(page, writer, label, cx, cy + 2, size * 0.78, ink)

    max_x = max(n["x"] for n in block["nodes"]) + r + 40
    max_y = max(n["y"] for n in block["nodes"]) + r + 40
    return blanks, (int(max_x), int(max_y))


def render_flow(
    page: Image.Image,
    block: dict,
    writer: Writer,
    origin: tuple[int, int],
    ink: tuple[int, int, int],
    size: float,
    rng: random.Random,
    answers: dict[str, str],
    reveal: set[str] | None = None,
) -> tuple[list[Blank], tuple[int, int]]:
    reveal = reveal or set()
    ox, oy = origin
    draw = ImageDraw.Draw(page)
    blanks: list[Blank] = []
    steps = block.get("steps", [])
    box_w = block.get("box_width", 520)
    box_h = int(size * 1.55)
    gap = int(size * 0.85)
    x = ox + 20
    y = oy
    for i, step in enumerate(steps):
        if isinstance(step, str):
            text, blank_id = step, None
        else:
            text, blank_id = step.get("text", ""), step.get("blank")
        box = (x, y, x + box_w, y + box_h)
        sketch_rect(draw, box, ink, width=2, rng=rng, fill=(252, 250, 245))
        if blank_id and blank_id not in reveal:
            blanks.append(
                {
                    "id": blank_id,
                    "kind": "flow_step",
                    "bbox": list(box),
                    "answer": answers.get(blank_id, text),
                    "hint": f"Step {i + 1}",
                }
            )
        else:
            frag, base, _ = writer.fragment(text, size=size * 0.7)
            paste_centered(page, frag, base, x + box_w / 2, y + box_h * 0.62, ink)
        if i < len(steps) - 1:
            ay0, ay1 = y + box_h + 4, y + box_h + gap - 4
            sketch_line(draw, (x + box_w / 2, ay0, x + box_w / 2, ay1), ink, width=2, rng=rng)
            # arrow head
            sketch_line(
                draw,
                (x + box_w / 2 - 8, ay1 - 10, x + box_w / 2, ay1),
                ink,
                width=2,
                rng=rng,
            )
            sketch_line(
                draw,
                (x + box_w / 2 + 8, ay1 - 10, x + box_w / 2, ay1),
                ink,
                width=2,
                rng=rng,
            )
        y += box_h + gap
    return blanks, (box_w + 40, y - oy)


def render_compare(
    page: Image.Image,
    block: dict,
    writer: Writer,
    origin: tuple[int, int],
    ink: tuple[int, int, int],
    size: float,
    rng: random.Random,
    answers: dict[str, str],
    reveal: set[str] | None = None,
) -> tuple[list[Blank], tuple[int, int]]:
    reveal = reveal or set()
    ox, oy = origin
    draw = ImageDraw.Draw(page)
    blanks: list[Blank] = []
    left, right = block.get("left", {}), block.get("right", {})
    col_w = block.get("col_width", 320)
    pad = 16
    row_h = int(size * 1.35)

    def col(meta: dict, x: int) -> int:
        title = meta.get("title", "")
        _label(page, writer, title, x + col_w / 2, oy + size * 0.6, size * 0.85, ink)
        y = oy + int(size * 1.4)
        for item in meta.get("items", []):
            if isinstance(item, str):
                text, blank_id = item, None
            else:
                text, blank_id = item.get("text", ""), item.get("blank")
            box = (x, y, x + col_w, y + row_h)
            sketch_rect(draw, box, ink, width=1, rng=rng)
            if blank_id and blank_id not in reveal:
                blanks.append(
                    {
                        "id": blank_id,
                        "kind": "compare",
                        "bbox": list(box),
                        "answer": answers.get(blank_id, text),
                        "hint": title,
                    }
                )
            else:
                frag, base, _ = writer.fragment(text, size=size * 0.65)
                paste_centered(page, frag, base, x + col_w / 2, y + row_h * 0.62, ink)
            y += row_h + 10
        return y

    y_l = col(left, ox)
    y_r = col(right, ox + col_w + pad * 2)
    mid_x = ox + col_w + pad
    sketch_line(draw, (mid_x, oy + size, mid_x, max(y_l, y_r) - 10), ink, width=2, rng=rng)
    return blanks, (col_w * 2 + pad * 2, max(y_l, y_r) - oy)


def render_conceptmap(
    page: Image.Image,
    block: dict,
    writer: Writer,
    origin: tuple[int, int],
    ink: tuple[int, int, int],
    size: float,
    rng: random.Random,
    answers: dict[str, str],
    reveal: set[str] | None = None,
) -> tuple[list[Blank], tuple[int, int]]:
    reveal = reveal or set()
    ox, oy = origin
    draw = ImageDraw.Draw(page)
    blanks: list[Blank] = []
    nodes = {n["id"]: n for n in block.get("nodes", [])}
    for e in block.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sketch_line(
            draw,
            (ox + a["x"], oy + a["y"], ox + b["x"], oy + b["y"]),
            ink,
            width=2,
            rng=rng,
        )
        if e.get("label"):
            mx = ox + (a["x"] + b["x"]) / 2
            my = oy + (a["y"] + b["y"]) / 2 - 12
            _label(page, writer, e["label"], mx, my, size * 0.55, ink)
    for n in block.get("nodes", []):
        cx, cy = ox + n["x"], oy + n["y"]
        tw = int(size * (0.55 * len(n.get("label", n["id"])) + 1.2))
        th = int(size * 1.2)
        box = (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)
        sketch_rect(draw, box, ink, width=2, rng=rng, fill=(252, 250, 245))
        blank_id = n.get("blank")
        label = n.get("label", n["id"])
        if blank_id and blank_id not in reveal:
            blanks.append(
                {
                    "id": blank_id,
                    "kind": "concept",
                    "bbox": [int(v) for v in box],
                    "answer": answers.get(blank_id, label),
                    "hint": "Concept",
                }
            )
        else:
            _label(page, writer, label, cx, cy + 2, size * 0.7, ink)
    max_x = max(n["x"] for n in block["nodes"]) + 80
    max_y = max(n["y"] for n in block["nodes"]) + 50
    return blanks, (int(max_x), int(max_y))


def render_table(
    page: Image.Image,
    block: dict,
    writer: Writer,
    origin: tuple[int, int],
    ink: tuple[int, int, int],
    size: float,
    rng: random.Random,
    answers: dict[str, str],
    reveal: set[str] | None = None,
) -> tuple[list[Blank], tuple[int, int]]:
    reveal = reveal or set()
    ox, oy = origin
    draw = ImageDraw.Draw(page)
    blanks: list[Blank] = []
    headers = block.get("headers", ["Destination", "Next hop", "Cost"])
    col_ws = block.get("col_widths", [180, 160, 120])
    row_h = int(size * 1.4)
    x = ox
    # header
    for h, w in zip(headers, col_ws):
        sketch_rect(draw, (x, oy, x + w, oy + row_h), ink, width=2, rng=rng, fill=(240, 236, 228))
        _label(page, writer, h, x + w / 2, oy + row_h * 0.62, size * 0.65, ink)
        x += w
    y = oy + row_h
    for row in block.get("rows", []):
        cells = [
            ("dest", row.get("dest", ""), None),
            ("hop", row.get("hop"), row.get("blank_hop")),
            ("cost", row.get("cost"), row.get("blank_cost")),
        ]
        x = ox
        for (kind, value, blank_id), w in zip(cells, col_ws):
            box = (x, y, x + w, y + row_h)
            sketch_rect(draw, box, ink, width=1, rng=rng)
            show = blank_id is None or blank_id in reveal
            if blank_id and not show:
                blanks.append(
                    {
                        "id": blank_id,
                        "kind": f"table_{kind}",
                        "bbox": list(box),
                        "answer": answers.get(blank_id, str(value) if value is not None else ""),
                        "hint": f"{row.get('dest', '')} {kind}",
                    }
                )
            else:
                text = answers.get(blank_id, str(value) if value is not None else "") if blank_id else str(value)
                if text:
                    _label(page, writer, text, x + w / 2, y + row_h * 0.62, size * 0.7, ink)
            x += w
        y += row_h
    return blanks, (sum(col_ws), y - oy)
