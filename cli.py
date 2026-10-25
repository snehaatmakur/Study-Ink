"""Command line for the handwriting engine.

  python cli.py template --out template.png                       # 1) make the one-time sheet
  python cli.py extract --image filled.png --rows template_rows.json --out profiles/me   # 2) learn the handwriting
  python cli.py extract --image abc.png --rows letters --out profiles/me                 #    (a-z / A-Z only sample)
  python cli.py write --profile profiles/me --text "Hello R3!" --out page.png            # 3) render text
"""
import argparse
import json

from handwriting import Writer, add_fallbacks, extract_font, HandwritingFont
from handwriting.review import contact_sheet
from handwriting.template import make_template


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template"); t.add_argument("--out", default="template.png"); t.add_argument("--reps", type=int, default=2)
    e = sub.add_parser("extract"); e.add_argument("--image", required=True); e.add_argument("--rows", required=True)
    e.add_argument("--out", required=True); e.add_argument("--debug", default=None)
    e.add_argument("--sheet", default=None, help="write a contact sheet of every glyph variant to review")
    e.add_argument("--drop", default="", help="comma list: a whole char (e.g. →) or one variant (e.g. i:1)")
    w = sub.add_parser("write"); w.add_argument("--profile", required=True); w.add_argument("--text", required=True)
    w.add_argument("--out", default="page.png"); w.add_argument("--ink", default="#1a1a1f")
    w.add_argument("--paper", default="lined", choices=["none", "white", "lined", "grid"])
    w.add_argument("--size", type=float, default=56); w.add_argument("--width", type=int, default=1500)
    w.add_argument("--neatness", type=float, default=0.5); w.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    if a.cmd == "template":
        rows = make_template(a.out, a.out.rsplit(".", 1)[0] + "_rows.json", letter_reps=a.reps)
        print(f"wrote {a.out} with {len(rows)} rows")
    elif a.cmd == "extract":
        if a.rows == "letters":
            rows = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        else:
            rows = json.load(open(a.rows))["rows"]
        font = extract_font(a.image, rows, debug_path=a.debug)
        for tok in [t for t in a.drop.split(",") if t]:
            if len(tok) >= 3 and tok[1] == ":" and tok[2:].isdigit():
                ch, idx = tok[0], int(tok[2:])
                if len(font.glyphs.get(ch, [])) > idx:
                    font.glyphs[ch].pop(idx)
            else:
                font.glyphs.pop(tok, None)
        if a.sheet:
            contact_sheet(font, a.sheet)
        added = add_fallbacks(font)
        font.save(a.out)
        print(f"saved profile to {a.out}; stand-in glyphs used for: {''.join(added) or 'none'}")
    elif a.cmd == "write":
        font = HandwritingFont.load(a.profile)
        img = Writer(font, ink=hex_to_rgb(a.ink), seed=a.seed, neatness=a.neatness).write(
            a.text.replace("\\n", "\n"), width=a.width, size=a.size, paper=a.paper)
        img.save(a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
