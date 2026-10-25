import json, numpy as np
from PIL import Image
from handwriting import HandwritingFont, extract_font
from handwriting.template import make_template

font = HandwritingFont.load("profiles/trisha")           # stand-in "student" (her letters + fallback symbols)
rows = make_template("t_blank.png", "t_rows.json")
img = np.array(Image.open("t_blank.png").convert("RGB")).astype(np.float32)
cell_w, row_h = 84, 120
for ri, row in enumerate(rows):
    for ci, ch in enumerate(row):
        v = font.variants(ch)
        if not v: print("no glyph for", repr(ch)); continue
        g = v[0]; a = np.asarray(g.alpha, np.float32) / 255
        x = ci * cell_w + (cell_w - a.shape[1]) // 2
        y = ri * row_h + 86 - g.base
        reg = img[y:y + a.shape[0], x:x + a.shape[1]]
        img[y:y + a.shape[0], x:x + a.shape[1]] = reg * (1 - a[..., None]) + 15 * a[..., None]
Image.fromarray(img.astype(np.uint8)).save("t_filled.png")

f2 = extract_font("t_filled.png", rows, debug_path="t_debug.png")
missing = [c for c in set("".join(rows)) if not f2.has(c)]
counts = {c: len(f2.variants(c)) for c in "aA0x.,"}
print("rows:", len(rows), "| missing chars:", missing, "| variants:", counts)
print("xh/cap/stroke:", round(f2.xh,1), round(f2.cap_h,1), round(f2.stroke,1), "(original", font.xh, font.cap_h, round(font.stroke,1), ")")
