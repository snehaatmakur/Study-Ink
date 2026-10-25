"""Render typed text as handwriting using a student's HandwritingFont."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image

from .font import Glyph, HandwritingFont

CHAR_MAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "\u00a0": " "}
SCRIPT_CHARS = {
    **{ch: (str(i), "sub") for i, ch in enumerate("₀₁₂₃₄₅₆₇₈₉")},
    **{ch: (str(i), "super") for i, ch in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
    "₊": ("+", "sub"), "₋": ("-", "sub"),
    "⁺": ("+", "super"), "⁻": ("-", "super"), "−": ("-", "super"),
}


class Writer:
    def __init__(self, font: HandwritingFont, ink=(25, 25, 30), seed: int | None = 7, neatness: float = 0.5):
        """neatness: 0 = loose/natural, 1 = very uniform. Controls how much random variation is added."""
        self.font = font
        self.ink = ink
        self.rng = random.Random(seed)
        self.jit = max(0.0, 1.0 - neatness) * 1.6 + 0.15  # variation multiplier

    # ------------------------------------------------------------------ words
    def _pick(self, ch: str, last: Glyph | None) -> Glyph | None:
        vs = self.font.variants(CHAR_MAP.get(ch, ch))
        if not vs:
            return None
        if len(vs) > 1 and last is not None:
            vs = [v for v in vs if v is not last] or vs
        return self.rng.choice(vs)

    def _word(self, word: str, size: float):
        """Return (alpha array, baseline_y, width) for one word."""
        r, j = self.rng, self.jit
        s0 = (0.48 * size) / self.font.xh
        A, D = int(size * 1.5), int(size * 0.7)
        H = A + D
        parts, x, last = [], 6.0, None
        for i, ch in enumerate(word):
            script = SCRIPT_CHARS.get(ch)
            glyph_char = script[0] if script else ch
            g = self._pick(glyph_char, last)
            if g is None:
                continue
            last = g
            script_kind = script[1] if script else ""
            sc = s0 * (0.68 if script_kind else 1.0) * (1 + r.uniform(-0.05, 0.05) * j)
            w, h = max(1, int(g.alpha.width * sc)), max(1, int(g.alpha.height * sc))
            im = g.alpha.resize((w, h), Image.LANCZOS)
            ang = r.uniform(-1.8, 1.8) * j
            if abs(ang) > 0.1:
                im = im.rotate(ang, expand=True, resample=Image.BICUBIC, fillcolor=0)
            wob = r.uniform(-0.03, 0.03) * size * j + math.sin(i * 0.9 + r.random()) * 0.012 * size * j
            baseline = A - size * 0.28 if script_kind == "super" else A + size * 0.14 if script_kind == "sub" else A
            top = baseline - g.base * sc + wob - (im.height - h) / 2
            opacity = r.uniform(0.86, 1.0) if j > 0.2 else 1.0
            parts.append((im, x - (im.width - w) / 2, top, opacity))
            gap = size * (0.045 + r.uniform(-0.02, 0.04) * j)
            x += w + gap
        W = int(x + 8)
        canvas = np.zeros((H, W), np.float32)
        for im, px, py, op in parts:
            arr = np.asarray(im, dtype=np.float32) / 255.0 * op
            ix, iy = int(round(px)), int(round(py))
            x0, y0 = max(ix, 0), max(iy, 0)
            x1, y1 = min(ix + arr.shape[1], W), min(iy + arr.shape[0], H)
            if x1 > x0 and y1 > y0:
                sub = arr[y0 - iy:y1 - iy, x0 - ix:x1 - ix]
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)
        return canvas, A, W

    def fragment(self, text: str, size: float = 56) -> tuple[np.ndarray, int, int]:
        """Render a short string (without wrapping) to an alpha mask and baseline."""
        text = "".join(CHAR_MAP.get(ch, ch) for ch in text)
        if not text.strip():
            return np.zeros((int(size * 2.2), max(1, int(size * 0.2))), np.float32), int(size * 1.5), max(1, int(size * 0.2))
        space = size * 0.36
        parts = []
        total_width = 0.0
        for word in text.split(" "):
            if not word:
                continue
            alpha, baseline, width = self._word(word, size)
            gap = space * self.rng.uniform(0.85, 1.25) if parts else 0.0
            parts.append((alpha, baseline, width, gap))
            total_width += gap + width
        if not parts:
            return np.zeros((int(size * 2.2), 1), np.float32), int(size * 1.5), 1
        ascent, descent = int(size * 1.5), int(size * 0.7)
        height, width = ascent + descent, int(total_width + 8)
        canvas = np.zeros((height, width), np.float32)
        x = 0.0
        for alpha, baseline, part_width, gap in parts:
            x += gap
            ix, iy = int(round(x)), int(round(ascent - baseline))
            h, w = alpha.shape
            x0, y0 = max(ix, 0), max(iy, 0)
            x1, y1 = min(ix + w, width), min(iy + h, height)
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], alpha[y0 - iy:y1 - iy, x0 - ix:x1 - ix])
            x += part_width
        return canvas, ascent, width

    # ------------------------------------------------------------------ paragraphs
    def write(self, text: str, width: int = 1400, size: float = 56, line_height: float = 1.55,
              margin: int = 50, paper: str = "none", align: str = "left") -> Image.Image:
        """paper: 'none' (transparent), 'white', 'lined', 'grid'."""
        r, j = self.rng, self.jit
        text = "".join(CHAR_MAP.get(c, c) for c in text)
        max_w = width - 2 * margin
        space = size * 0.36
        lines: list[list[tuple[np.ndarray, int, int]]] = []
        for para in text.split("\n"):
            cur, cur_w = [], 0.0
            for word in para.split(" "):
                if not word:
                    continue
                arr, base, w = self._word(word, size)
                add = w + (space * r.uniform(0.85, 1.25) if cur else 0)
                if cur and cur_w + add > max_w:
                    lines.append(cur)
                    cur, cur_w, add = [], 0.0, w
                cur.append((arr, base, w, add - w))
                cur_w += add
            lines.append(cur)

        lh = size * line_height
        H = int(margin * 2 + size * 0.9 + lh * len(lines))
        page = np.zeros((H, width), np.float32)
        for li, line in enumerate(lines):
            base_y = margin + size * 1.0 + li * lh
            slope = r.uniform(-0.006, 0.006) * j
            total = sum(w + sp for _, _, w, sp in line)
            x = margin + ((max_w - total) / 2 if align == "center" else 0)
            for arr, base, w, sp in line:
                x += sp
                y = base_y - base + slope * (x - margin) + r.uniform(-0.01, 0.01) * size * j
                ix, iy = int(round(x)), int(round(y))
                h_, w_ = arr.shape
                x0, y0 = max(ix, 0), max(iy, 0)
                x1, y1 = min(ix + w_, width), min(iy + h_, H)
                if x1 > x0 and y1 > y0:
                    page[y0:y1, x0:x1] = np.maximum(page[y0:y1, x0:x1], arr[y0 - iy:y1 - iy, x0 - ix:x1 - ix])
                x += w

        return self._compose(page, paper, margin, size, lh)

    def _compose(self, a: np.ndarray, paper: str, margin: int, size: float, lh: float) -> Image.Image:
        H, W = a.shape
        ink = np.array(self.ink, np.float32)
        if paper == "none":
            rgba = np.zeros((H, W, 4), np.uint8)
            rgba[..., :3] = ink.astype(np.uint8)
            rgba[..., 3] = (a * 255).astype(np.uint8)
            return Image.fromarray(rgba, "RGBA")
        bg = np.full((H, W, 3), 250, np.float32)
        if paper == "lined":
            y = margin + size * 1.0 + size * 0.22
            while y < H - margin * 0.5:
                bg[int(y):int(y) + 2, :] = (200, 212, 232)
                y += lh
        elif paper == "grid":
            step = int(size * 0.9)
            bg[::step, :] = (205, 215, 232)
            bg[:, ::step] = (205, 215, 232)
        out = bg * (1 - a[..., None]) + ink * a[..., None]
        return Image.fromarray(out.astype(np.uint8), "RGB")
