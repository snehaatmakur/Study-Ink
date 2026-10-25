"""Turn a photo/screenshot of someone's handwriting (rows of characters) into a HandwritingFont.

Assumptions about the sample image:
  * Characters are written in left-to-right rows, in the same order as the `rows` strings you pass in.
  * Characters in a row do not touch each other (pieces of one character, like the dot of an "i",
    are fine and get merged automatically).
  * Paper, grid lines and guides are light. Ink is clearly darker (any pen color works).
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .font import Glyph, HandwritingFont

# characters that sit on the baseline (used to measure where the baseline is in each row)
BASELINE_ANCHORS = set("acehimnorstuvwxzABCDEFHIKLMNOPRSTUVWXYZ0123456789")
X_HEIGHT_CHARS = set("acemnorsuvwxz")
CAP_CHARS = set("EFHILTZ")


class ExtractionError(Exception):
    pass


def ink_alpha(img: Image.Image) -> np.ndarray:
    """0..1 'how much ink' per pixel. Uses the darkest channel so blue/green/red pens work
    but light-blue grid lines (min channel ~180+) do not."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    mn = rgb.min(axis=2)
    return np.clip((170.0 - mn) / 100.0, 0.0, 1.0)


def _components(alpha: np.ndarray, min_area: int = 6):
    mask = alpha > 0.5
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        area = int((lab[sl] == i).sum())
        if area >= min_area:
            comps.append(dict(id=i, x0=xs.start, x1=xs.stop, y0=ys.start, y1=ys.stop, area=area))
    return lab, comps


def _group_rows(comps):
    rows, cur, cur_y1 = [], [], -1
    for c in sorted(comps, key=lambda c: c["y0"]):
        if cur and c["y0"] >= cur_y1:
            rows.append(cur)
            cur, cur_y1 = [], -1
        cur.append(c)
        cur_y1 = max(cur_y1, c["y1"])
    if cur:
        rows.append(cur)
    # drop stray specks (a "row" with far less ink than the others)
    if len(rows) > 1:
        areas = [sum(c["area"] for c in r) for r in rows]
        med = float(np.median(areas))
        rows = [r for r, a in zip(rows, areas) if a > 0.1 * med]
    return rows


def _cluster_row(row_comps, merge_frac):
    clusters = []
    for c in sorted(row_comps, key=lambda c: c["x0"]):
        if clusters:
            cl = clusters[-1]
            overlap = min(cl["x1"], c["x1"]) - max(cl["x0"], c["x0"])
            narrow = min(cl["x1"] - cl["x0"], c["x1"] - c["x0"])
            if overlap > 0 and overlap >= merge_frac * narrow:
                cl["x0"], cl["x1"] = min(cl["x0"], c["x0"]), max(cl["x1"], c["x1"])
                cl["y0"], cl["y1"] = min(cl["y0"], c["y0"]), max(cl["y1"], c["y1"])
                cl["ids"].append(c["id"])
                continue
        clusters.append(dict(x0=c["x0"], x1=c["x1"], y0=c["y0"], y1=c["y1"], ids=[c["id"]]))
    return clusters


def _merge_closest(clusters, expected):
    """Characters made of side-by-side pieces (like the two ticks of ") come out as extra clusters.
    Pieces of one character sit closer together than separate characters do, so merge the closest pairs."""
    cl = [dict(c, ids=list(c["ids"])) for c in clusters]
    while len(cl) > expected:
        gaps = [cl[i + 1]["x0"] - cl[i]["x1"] for i in range(len(cl) - 1)]
        i = int(np.argmin(gaps))
        a, b = cl[i], cl.pop(i + 1)
        a["x0"], a["x1"] = min(a["x0"], b["x0"]), max(a["x1"], b["x1"])
        a["y0"], a["y1"] = min(a["y0"], b["y0"]), max(a["y1"], b["y1"])
        a["ids"] += b["ids"]
    return cl


def _cluster_row_to_count(row_comps, expected):
    for frac in (0.4, 0.3, 0.2, 0.5, 0.65, 0.8, 0.1):
        cl = _cluster_row(row_comps, frac)
        if len(cl) == expected:
            return cl
    base = _cluster_row(row_comps, 0.4)
    if len(base) > expected:
        return _merge_closest(base, expected)
    return None


def _crop(alpha, lab, cl, pad=2):
    H, W = alpha.shape
    X0, X1 = max(0, cl["x0"] - pad), min(W, cl["x1"] + pad)
    Y0, Y1 = max(0, cl["y0"] - pad), min(H, cl["y1"] + pad)
    m = np.isin(lab[Y0:Y1, X0:X1], cl["ids"])
    m = ndi.binary_dilation(m, iterations=2)  # recover soft anti-aliased edges
    a = alpha[Y0:Y1, X0:X1] * m
    return Image.fromarray((a * 255).astype(np.uint8), "L"), Y0


def estimate_stroke(glyphs: dict[str, list[Glyph]]) -> float:
    """Pen thickness ~ 2 * area / perimeter for a thin stroke."""
    vals = []
    for ch, vs in glyphs.items():
        if not ch.isalpha():
            continue
        for g in vs:
            m = (np.asarray(g.alpha) > 127).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            per = sum(cv2.arcLength(c, True) for c in cnts)
            if per > 0:
                vals.append(2.0 * m.sum() / per)
    return float(np.median(vals)) if vals else 3.0


def extract_font(image, rows: list[str], debug_path: str | None = None) -> HandwritingFont:
    img = Image.open(image) if isinstance(image, str) else image
    alpha = ink_alpha(img)
    lab, comps = _components(alpha)
    row_groups = _group_rows(comps)
    if len(row_groups) != len(rows):
        raise ExtractionError(
            f"Found {len(row_groups)} rows of writing but expected {len(rows)}. "
            "Check that rows do not touch and that nothing else is drawn on the page."
        )

    glyphs: dict[str, list[Glyph]] = {}
    boxes = []
    heights_x, heights_cap = [], []
    for ri, (group, chars) in enumerate(zip(row_groups, rows)):
        clusters = _cluster_row_to_count(group, len(chars))
        if clusters is None:
            raise ExtractionError(
                f"Row {ri + 1}: expected {len(chars)} characters ({chars!r}) but could not "
                "separate them. Leave a little more space between characters."
            )
        anchors = [cl["y1"] for cl, ch in zip(clusters, chars) if ch in BASELINE_ANCHORS]
        if len(anchors) < 2:
            raise ExtractionError(f"Row {ri + 1} has no letters that sit on the baseline to measure from.")
        baseline = float(np.median(anchors))
        for cl, ch in zip(clusters, chars):
            a, y_top = _crop(alpha, lab, cl)
            bl = baseline
            # Letters that belong on the line sit on it, even if they were written a few px high or low
            # in their cell. (The renderer adds natural wobble back.)
            if ch in BASELINE_ANCHORS:
                bl = float(cl["y1"])
            base = int(round(bl - y_top))
            glyphs.setdefault(ch, []).append(Glyph(a, base))
            boxes.append((cl, ch, bl))
            if ch in X_HEIGHT_CHARS:
                heights_x.append(bl - cl["y0"])
            if ch in CAP_CHARS:
                heights_cap.append(bl - cl["y0"])

    stroke = estimate_stroke(glyphs)
    xh = float(np.median(heights_x)) if heights_x else (0.68 * float(np.median(heights_cap)) if heights_cap else 20.0)
    cap_h = float(np.median(heights_cap)) if heights_cap else xh * 1.5

    if debug_path:
        dbg = img.convert("RGB").copy()
        d = ImageDraw.Draw(dbg)
        for cl, ch, bl in boxes:
            d.rectangle([cl["x0"], cl["y0"], cl["x1"], cl["y1"]], outline=(255, 0, 0), width=1)
            d.line([cl["x0"] - 3, bl, cl["x1"] + 3, bl], fill=(0, 160, 0), width=1)
        dbg.save(debug_path)

    return HandwritingFont(glyphs, xh=xh, cap_h=cap_h, stroke=stroke)


def suspicious_variants(font: HandwritingFont, ratio: float = 1.5) -> list[str]:
    """Flag characters whose variants differ a lot in size (usually a bad or mislabeled cell)."""
    flags = []
    for ch, vs in font.glyphs.items():
        if len(vs) < 2:
            continue
        hs = [v.alpha.height for v in vs]
        ws = [v.alpha.width for v in vs]
        if max(hs) / max(1, min(hs)) > ratio or max(ws) / max(1, min(ws)) > ratio * 1.3:
            flags.append(f"{ch!r}: variant heights {hs}, widths {ws}")
    return flags
