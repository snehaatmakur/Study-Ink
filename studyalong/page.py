"""Compose a full Fill in the Ink (Study Companion) page from a lesson JSON document."""
from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from handwriting import HandwritingFont, Writer, add_fallbacks

from . import diagrams
from .sketch import paste_alpha, sketch_line, sketch_rect
from .solver import answers_for_lesson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILE = os.path.join(ROOT, "profiles", "trisha_full")
LESSONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons")


@dataclass
class RenderedLesson:
    image: Image.Image
    blanks: list[dict[str, Any]] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    document: dict[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0

    def png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, "PNG")
        return buf.getvalue()

    def export_json(self) -> dict[str, Any]:
        return {
            "title": self.document.get("title"),
            "lesson_id": self.document.get("id"),
            "width": self.width,
            "height": self.height,
            "blanks": self.blanks,
            "answers": self.answers,
            "blocks": self.document.get("blocks", []),
            "integration": {
                "note": "Map block IDs and blank bboxes onto Notability objects later.",
                "format": "studyalong.v1",
            },
        }


def load_lesson(lesson_id: str) -> dict:
    path = os.path.join(LESSONS_DIR, f"{lesson_id}.json")
    with open(path) as f:
        return json.load(f)


def list_lessons() -> list[dict[str, str]]:
    out = []
    for name in sorted(os.listdir(LESSONS_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(LESSONS_DIR, name)) as f:
            data = json.load(f)
        out.append({"id": data.get("id", name[:-5]), "title": data.get("title", name), "subtitle": data.get("subtitle", "")})
    return out


def _load_font(profile: str | None = None) -> HandwritingFont:
    path = profile or DEFAULT_PROFILE
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    font = HandwritingFont.load(path)
    add_fallbacks(font)
    return font


def _paper(width: int, height: int, kind: str = "lined", size: float = 48, margin: int = 56) -> Image.Image:
    bg = np.full((height, width, 3), 250, np.uint8)
    if kind == "lined":
        y = margin + int(size * 1.2)
        lh = int(size * 1.55)
        while y < height - margin // 2:
            bg[y:y + 2, :] = (200, 212, 232)
            y += lh
    elif kind == "grid":
        step = int(size * 0.9)
        bg[::step, :] = (205, 215, 232)
        bg[:, ::step] = (205, 215, 232)
    return Image.fromarray(bg, "RGB")


def _draw_runs(
    page: Image.Image,
    writer: Writer,
    runs: list[dict],
    x: float,
    y_base: float,
    max_x: float,
    size: float,
    ink: tuple[int, int, int],
    answers: dict[str, str],
    reveal: set[str],
    blanks: list[dict],
    line_height: float,
    margin: int,
) -> tuple[float, float]:
    """Word-wrap runs with blanks. y_base is the baseline of the first line. Returns (x, y_base)."""
    space = size * 0.36
    draw = ImageDraw.Draw(page)

    def new_line(cur_x, cur_y):
        return margin, cur_y + size * line_height

    for run in runs:
        if "blank" in run:
            bid = run["blank"]
            ans = answers.get(bid, run.get("answer", ""))
            bw = int(run.get("width", max(size * 0.55 * max(len(ans), 4), size * 2.5)))
            bh = int(size * 0.9)
            if x + bw > max_x:
                x, y_base = new_line(x, y_base)
            box = (int(x), int(y_base - bh * 0.75), int(x + bw), int(y_base + bh * 0.15))
            if bid in reveal:
                frag, base, w = writer.fragment(ans, size=size)
                paste_alpha(page, frag, int(x), int(y_base - base), ink)
            else:
                # underline blank
                sketch_line(
                    draw,
                    (box[0], y_base + 2, box[2], y_base + 4),
                    ink,
                    width=2,
                    rng=writer.rng,
                )
                blanks.append(
                    {
                        "id": bid,
                        "kind": "text",
                        "bbox": list(box),
                        "answer": ans,
                        "hint": run.get("hint", "Fill in the blank"),
                    }
                )
            x += bw + space * 0.6
            continue

        text = run.get("t", "")
        for wi, word in enumerate(text.split(" ")):
            if word == "":
                continue
            frag, base, w = writer.fragment(word, size=size)
            gap = space if (wi > 0 or x > margin + 1) else 0
            if x + gap + w > max_x and x > margin + 1:
                x, y_base = new_line(x, y_base)
                gap = 0
            x += gap
            paste_alpha(page, frag, int(x), int(y_base - base), ink)
            x += w
    return x, y_base


def render_lesson(
    lesson: dict | str,
    *,
    profile: str | None = None,
    ink: tuple[int, int, int] = (12, 88, 42),
    size: float = 44,
    width: int = 1100,
    paper: str = "lined",
    seed: int = 7,
    neatness: float = 0.55,
    reveal: set[str] | frozenset[str] | None = None,
    mode: str = "study",  # study | key | redraw_ref
) -> RenderedLesson:
    if isinstance(lesson, str):
        lesson = load_lesson(lesson)
    reveal_set = set(reveal or [])
    if mode == "key":
        # reveal everything
        answers_preview = answers_for_lesson(lesson)
        reveal_set = set(answers_preview.keys())

    font = _load_font(profile)
    writer = Writer(font, ink=ink, seed=seed, neatness=neatness)
    answers = answers_for_lesson(lesson)
    rng = random.Random(seed + 99)

    margin = 56
    # first pass height estimate — generous, then crop
    est_h = 3200
    page = _paper(width, est_h, paper, size=size, margin=margin)
    blanks: list[dict[str, Any]] = []
    y = margin + size * 0.3
    max_x = width - margin

    # title
    title = lesson.get("title", "Fill in the Ink")
    frag, base, _ = writer.fragment(title, size=size * 1.35)
    paste_alpha(page, frag, margin, int(y + size * 0.9 - base), ink)
    y += size * 1.8
    if lesson.get("subtitle"):
        frag, base, _ = writer.fragment(lesson["subtitle"], size=size * 0.85)
        paste_alpha(page, frag, margin, int(y + size * 0.7 - base), ink)
        y += size * 1.4

    for block in lesson.get("blocks", []):
        btype = block.get("type")
        y += size * 0.35
        if btype == "heading":
            frag, base, _ = writer.fragment(block.get("text", ""), size=size * 1.05)
            paste_alpha(page, frag, margin, int(y + size * 0.85 - base), ink)
            y += size * 1.5
        elif btype == "text":
            _, y_base = _draw_runs(
                page,
                writer,
                block.get("runs", []),
                margin,
                y + size,
                max_x,
                size,
                ink,
                answers,
                reveal_set,
                blanks,
                line_height=1.55,
                margin=margin,
            )
            y = y_base + size * 0.55
        elif btype == "graph":
            if mode == "redraw_ref" and block.get("id") == lesson.get("redraw_of"):
                # still draw reference lightly? for redraw mode we skip graph on main page
                pass
            b_list, (bw, bh) = diagrams.render_graph(
                page, block, writer, (margin, int(y)), ink, size, rng, answers, reveal_set
            )
            blanks.extend(b_list)
            y += bh + size * 0.4
        elif btype == "flow":
            b_list, (bw, bh) = diagrams.render_flow(
                page, block, writer, (margin, int(y)), ink, size, rng, answers, reveal_set
            )
            blanks.extend(b_list)
            y += bh + size * 0.3
        elif btype == "compare":
            b_list, (bw, bh) = diagrams.render_compare(
                page, block, writer, (margin, int(y)), ink, size, rng, answers, reveal_set
            )
            blanks.extend(b_list)
            y += bh + size * 0.3
        elif btype == "conceptmap":
            b_list, (bw, bh) = diagrams.render_conceptmap(
                page, block, writer, (margin, int(y)), ink, size, rng, answers, reveal_set
            )
            blanks.extend(b_list)
            y += bh + size * 0.3
        elif btype == "table":
            b_list, (bw, bh) = diagrams.render_table(
                page, block, writer, (margin, int(y)), ink, size, rng, answers, reveal_set
            )
            blanks.extend(b_list)
            y += bh + size * 0.4
        elif btype == "question":
            prompt = block.get("prompt", "")
            bid = block.get("blank") or block["id"]
            frag, base, pw = writer.fragment(prompt + " ", size=size)
            paste_alpha(page, frag, margin, int(y + size - base), ink)
            ans = answers.get(bid, block.get("answer", ""))
            bw = int(block.get("width", max(size * 0.55 * max(len(ans), 3), size * 3)))
            bx = margin + pw
            if bx + bw > max_x:
                y += size * 1.55
                bx = margin
            box = (int(bx), int(y + size - size * 0.75), int(bx + bw), int(y + size + size * 0.15))
            if bid in reveal_set:
                frag, base, _ = writer.fragment(ans, size=size)
                paste_alpha(page, frag, int(bx), int(y + size - base), ink)
            else:
                draw = ImageDraw.Draw(page)
                sketch_line(draw, (box[0], y + size + 2, box[2], y + size + 4), ink, width=2, rng=rng)
                blanks.append(
                    {
                        "id": bid,
                        "kind": "question",
                        "bbox": list(box),
                        "answer": ans,
                        "hint": block.get("hint", "Think about the shortest path"),
                    }
                )
            y += size * 1.7
        elif btype == "redraw":
            prompt = block.get("prompt", "Redraw the diagram from memory")
            frag, base, _ = writer.fragment(prompt, size=size * 0.9)
            paste_alpha(page, frag, margin, int(y + size * 0.8 - base), ink)
            y += size * 1.3
            rh = int(block.get("height", 360))
            rw = int(block.get("width", width - 2 * margin))
            box = (margin, int(y), margin + rw, int(y + rh))
            draw = ImageDraw.Draw(page)
            sketch_rect(draw, box, ink, width=2, rng=rng)
            blanks.append(
                {
                    "id": block["id"],
                    "kind": "redraw",
                    "bbox": list(box),
                    "answer": "",
                    "hint": "Sketch, then compare to the reference",
                    "ref_block": block.get("ref"),
                }
            )
            y += rh + size * 0.5

    height = min(int(y + margin), est_h)
    page = page.crop((0, 0, width, height))
    # filter blanks to those still on page
    doc = {
        "id": lesson.get("id"),
        "title": lesson.get("title"),
        "subtitle": lesson.get("subtitle"),
        "blocks": lesson.get("blocks", []),
    }
    return RenderedLesson(
        image=page,
        blanks=blanks,
        answers={b["id"]: b["answer"] for b in blanks if b.get("answer") != ""},
        document=doc,
        width=width,
        height=height,
    )


def render_redraw_reference(
    lesson: dict | str,
    ref_block_id: str,
    **kwargs,
) -> Image.Image:
    if isinstance(lesson, str):
        lesson = load_lesson(lesson)
    block = next(b for b in lesson["blocks"] if b["id"] == ref_block_id)
    font = _load_font(kwargs.get("profile"))
    ink = kwargs.get("ink", (12, 88, 42))
    size = kwargs.get("size", 40)
    seed = kwargs.get("seed", 7)
    writer = Writer(font, ink=ink, seed=seed, neatness=kwargs.get("neatness", 0.55))
    answers = answers_for_lesson(lesson)
    rng = random.Random(seed)
    # sized canvas
    if block["type"] == "graph":
        max_x = max(n["x"] for n in block["nodes"]) + 80
        max_y = max(n["y"] for n in block["nodes"]) + 80
    else:
        max_x, max_y = 700, 400
    page = Image.new("RGB", (int(max_x + 40), int(max_y + 40)), (250, 250, 248))
    if block["type"] == "graph":
        diagrams.render_graph(page, block, writer, (20, 20), ink, size, rng, answers, set(answers))
    return page
