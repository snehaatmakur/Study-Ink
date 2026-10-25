"""One-time calibration sheet. The student fills each labeled cell with the Pencil (or pen), then we extract.

Rows that contain punctuation start with the anchor letters "xn": they sit on the baseline, which is how
the extractor measures where the baseline is for rows that have no other baseline letters."""
from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageFont

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = LOWER.upper()
DIGITS = "0123456789"
PUNCT_1 = "xn" + ".,;:!?'\"-()"
PUNCT_2 = "xn" + "[]/+=<>*%&#@_•→←~"


def template_rows(letter_reps: int = 2) -> list[str]:
    return [LOWER] * letter_reps + [UPPER] * letter_reps + [DIGITS] * letter_reps + [PUNCT_1, PUNCT_2]


def make_template(png_path: str, rows_json_path: str, letter_reps: int = 2, cell_w: int = 84, row_h: int = 120):
    rows = template_rows(letter_reps)
    cols = max(len(r) for r in rows)
    img = Image.new("RGB", (cols * cell_w, len(rows) * row_h), "white")
    d = ImageDraw.Draw(img)
    try:
        label_font = ImageFont.load_default(size=15)
    except TypeError:
        label_font = ImageFont.load_default()
    box, guide, label = (190, 205, 235), (180, 195, 230), (195, 205, 235)  # all light: ignored as "not ink"
    for ri, row in enumerate(rows):
        y0 = ri * row_h
        d.line([0, y0 + 86, len(row) * cell_w, y0 + 86], fill=guide, width=2)  # baseline guide
        for ci, ch in enumerate(row):
            x0 = ci * cell_w
            d.rectangle([x0, y0, x0 + cell_w, y0 + row_h], outline=box, width=2)
            d.text((x0 + 6, y0 + 4), ch, fill=label, font=label_font)
    img.save(png_path)
    with open(rows_json_path, "w") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False)
    return rows
