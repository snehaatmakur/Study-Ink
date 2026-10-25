"""Stand-in glyphs (single-stroke vector shapes) for characters the student has not written yet.
They are scaled to the student's own letter size and pen thickness so the demo still works before
the student fills in digits and punctuation. Replace them by having the student complete the template."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .font import Glyph, HandwritingFont

FALLBACK_CHARS = list("0123456789.,;:!?'\"-()[]{}/\\+=<>*%&#@_~$^|") + ["•", "→", "←"]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SS = 4  # supersampling for smooth edges


def _trim(canvas: np.ndarray, baseline_y: float):
    ys, xs = np.where(canvas > 8)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return canvas[y0:y1, x0:x1], baseline_y - y0


def _draw(ch: str, cap_h: float, stroke: float):
    ss = _SS
    h_ref = cv2.getTextSize("H", _FONT, 1.0, 1)[0][1]
    scale = (cap_h * ss) / h_ref
    thick = max(1, int(round(stroke * ss)))
    pad = int(cap_h * ss * 0.6)
    size = int(cap_h * ss * 3)
    canvas = np.zeros((size, size), np.uint8)
    base_y = pad + int(cap_h * ss * 1.2)
    ox = pad

    if ch == "•":
        r = int(cap_h * ss * 0.12)
        cv2.circle(canvas, (ox + r, base_y - int(cap_h * ss * 0.4)), r + thick // 2, 255, -1, cv2.LINE_AA)
    elif ch in "→←":
        L = int(cap_h * ss * 1.1)
        y = base_y - int(cap_h * ss * 0.4)
        a, b = (ox, ox + L) if ch == "→" else (ox + L, ox)
        cv2.arrowedLine(canvas, (a, y), (b, y), 255, thick, cv2.LINE_AA, tipLength=0.28)
    elif ch == "^":
        # caret: small handwritten-looking peak above the baseline
        y = base_y - int(cap_h * ss * 0.95)
        mid = ox + int(cap_h * ss * 0.35)
        half = int(cap_h * ss * 0.35)
        cv2.line(canvas, (mid - half, base_y - int(cap_h * ss * 0.35)), (mid, y), 255, thick, cv2.LINE_AA)
        cv2.line(canvas, (mid, y), (mid + half, base_y - int(cap_h * ss * 0.35)), 255, thick, cv2.LINE_AA)
    elif ch == "|":
        cv2.line(canvas, (ox + thick, base_y - int(cap_h * ss)), (ox + thick, base_y), 255, thick, cv2.LINE_AA)
    elif ch in "{}":
        # curly braces via small polylines (Hershey font braces look too printed)
        h = int(cap_h * ss)
        x = ox + int(h * 0.35)
        pts_l = np.array([
            [x + int(h * 0.25), base_y - h],
            [x, base_y - int(h * 0.75)],
            [x, base_y - int(h * 0.55)],
            [x - int(h * 0.2), base_y - int(h * 0.5)],
            [x, base_y - int(h * 0.45)],
            [x, base_y - int(h * 0.25)],
            [x + int(h * 0.25), base_y],
        ], np.int32)
        pts = pts_l if ch == "{" else pts_l * np.array([[-1, 1]]) + np.array([[2 * x, 0]])
        cv2.polylines(canvas, [pts], False, 255, thick, cv2.LINE_AA)
    else:
        text = ch
        (w, h), bl = cv2.getTextSize(text, _FONT, scale, thick)
        cv2.putText(canvas, text, (ox, base_y), _FONT, scale, 255, thick, cv2.LINE_AA)

    small = cv2.resize(canvas, (canvas.shape[1] // ss, canvas.shape[0] // ss), interpolation=cv2.INTER_AREA)
    t = _trim(small, base_y / ss)
    if t is None:
        return None
    arr, base = t
    arr = np.pad(arr, 2)
    return Glyph(Image.fromarray(arr, "L"), int(round(base + 2)))


def add_fallbacks(font: HandwritingFont) -> list[str]:
    """Fill in any missing characters. Returns the list of characters that were synthesized."""
    added = []
    for ch in FALLBACK_CHARS:
        if font.has(ch):
            continue
        g = _draw(ch, cap_h=font.cap_h * 0.95, stroke=font.stroke)
        if g is not None:
            font.glyphs[ch] = [g]
            added.append(ch)
    return added
