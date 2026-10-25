"""A personal handwriting 'font': glyph alpha masks + baseline metrics, saved as PNGs + JSON."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PIL import Image


@dataclass
class Glyph:
    alpha: Image.Image  # mode "L", ink = 255, tightly cropped (plus a small pad)
    base: int           # y of the baseline, measured from the top of `alpha`


class HandwritingFont:
    def __init__(self, glyphs: dict[str, list[Glyph]], xh: float, cap_h: float, stroke: float):
        self.glyphs = glyphs
        self.xh = xh          # x-height in px (distance from baseline up to top of x-height letters)
        self.cap_h = cap_h    # capital height in px
        self.stroke = stroke  # average pen thickness in px

    def has(self, ch: str) -> bool:
        return bool(self.glyphs.get(ch))

    def variants(self, ch: str) -> list[Glyph]:
        return self.glyphs.get(ch, [])

    # ---- persistence -------------------------------------------------
    def save(self, folder: str) -> None:
        os.makedirs(os.path.join(folder, "glyphs"), exist_ok=True)
        meta = {"xh": self.xh, "cap_h": self.cap_h, "stroke": self.stroke, "glyphs": {}}
        for ch, variants in self.glyphs.items():
            entries = []
            for i, g in enumerate(variants):
                name = f"U{ord(ch):04X}_{i}.png"
                g.alpha.save(os.path.join(folder, "glyphs", name))
                entries.append({"file": name, "base": g.base})
            meta["glyphs"][f"U{ord(ch):04X}"] = entries
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, folder: str) -> "HandwritingFont":
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        glyphs: dict[str, list[Glyph]] = {}
        for key, entries in meta["glyphs"].items():
            ch = chr(int(key[1:], 16))
            glyphs[ch] = [
                Glyph(Image.open(os.path.join(folder, "glyphs", e["file"])).convert("L"), e["base"])
                for e in entries
            ]
        return cls(glyphs, meta["xh"], meta["cap_h"], meta["stroke"])
