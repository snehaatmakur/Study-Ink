"""Contact sheet of every extracted glyph variant, labeled char:index, so bad cells are easy to spot
(then re-run extract with --drop "i:1" etc.)."""
from PIL import Image, ImageDraw, ImageFont

from .font import HandwritingFont


def contact_sheet(font: HandwritingFont, path: str, cols: int = 16, cell: int = 96) -> None:
    items = [(ch, i, g) for ch, vs in font.glyphs.items() for i, g in enumerate(vs)]
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    d = ImageDraw.Draw(sheet)
    try:
        lf = ImageFont.load_default(size=13)
    except TypeError:
        lf = ImageFont.load_default()
    for n, (ch, i, g) in enumerate(items):
        x0, y0 = (n % cols) * cell, (n // cols) * cell
        d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(220, 225, 240))
        s = min(1.0, (cell - 30) / max(g.alpha.width, g.alpha.height))
        im = g.alpha.resize((max(1, int(g.alpha.width * s)), max(1, int(g.alpha.height * s))))
        ink = Image.new("RGB", im.size, (20, 20, 20))
        sheet.paste(ink, (x0 + (cell - im.width) // 2, y0 + 6 + (cell - 30 - im.height) // 2), im)
        d.text((x0 + 6, y0 + cell - 18), f"{ch}:{i}", fill=(40, 90, 200), font=lf)
    sheet.save(path)
