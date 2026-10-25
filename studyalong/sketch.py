"""Sketchy (hand-drawn looking) vector primitives via Pillow — no image model involved."""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def _wobble_points(
    x0: float, y0: float, x1: float, y1: float, rng: random.Random, roughness: float = 1.0
) -> list[tuple[float, float]]:
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    n = max(2, int(length / 18))
    pts = []
    nx, ny = -dy / length, dx / length
    for i in range(n + 1):
        t = i / n
        # slight bow + noise so edges don't look CAD-perfect
        bow = math.sin(t * math.pi) * roughness * rng.uniform(1.2, 2.8)
        jx = rng.uniform(-1.1, 1.1) * roughness
        jy = rng.uniform(-1.1, 1.1) * roughness
        pts.append((x0 + dx * t + nx * bow + jx, y0 + dy * t + ny * bow + jy))
    return pts


def sketch_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    fill,
    width: int = 2,
    rng: random.Random | None = None,
    roughness: float = 1.0,
) -> None:
    rng = rng or random.Random()
    x0, y0, x1, y1 = xy
    pts = _wobble_points(x0, y0, x1, y1, rng, roughness)
    # double stroke with slight offset for ink feel
    draw.line(pts, fill=fill, width=width, joint="curve")
    if width >= 2:
        offs = [(p[0] + rng.uniform(-0.6, 0.6), p[1] + rng.uniform(-0.6, 0.6)) for p in pts]
        draw.line(offs, fill=fill, width=max(1, width - 1), joint="curve")


def sketch_circle(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    r: float,
    fill=None,
    outline=None,
    width: int = 2,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random.Random()
    n = 28
    pts = []
    for i in range(n + 1):
        a = (i / n) * 2 * math.pi
        rr = r + rng.uniform(-1.4, 1.4)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    if fill is not None:
        draw.polygon(pts, fill=fill)
    if outline is not None:
        draw.line(pts, fill=outline, width=width, joint="curve")


def sketch_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    outline,
    width: int = 2,
    rng: random.Random | None = None,
    fill=None,
) -> None:
    rng = rng or random.Random()
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if fill is not None:
        draw.polygon(corners, fill=fill)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        sketch_line(draw, (a[0], a[1], b[0], b[1]), outline, width=width, rng=rng)


def paste_alpha(page: Image.Image, frag: np.ndarray, x: int, y: int, ink: tuple[int, int, int]) -> None:
    """Paste a float alpha mask onto an RGB(A) page at (x, y) top-left of the mask."""
    h, w = frag.shape
    if page.mode == "RGBA":
        overlay = np.zeros((h, w, 4), np.uint8)
        overlay[..., :3] = ink
        overlay[..., 3] = (np.clip(frag, 0, 1) * 255).astype(np.uint8)
        page.paste(Image.fromarray(overlay, "RGBA"), (x, y), Image.fromarray(overlay, "RGBA"))
    else:
        arr = np.asarray(page).astype(np.float32)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, page.width), min(y + h, page.height)
        if x1 <= x0 or y1 <= y0:
            return
        sub = frag[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
        ink_a = np.array(ink, np.float32)
        arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - sub) + ink_a * sub
        page.paste(Image.fromarray(arr.astype(np.uint8), page.mode))


def paste_centered(
    page: Image.Image,
    frag: np.ndarray,
    baseline: int,
    cx: float,
    cy: float,
    ink: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Paste so the glyph baseline sits on cy and the fragment is centered on cx. Returns bbox."""
    h, w = frag.shape
    x = int(round(cx - w / 2))
    y = int(round(cy - baseline))
    paste_alpha(page, frag, x, y, ink)
    return (x, y, x + w, y + h)
