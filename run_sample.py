import sys
from handwriting import extract_font, add_fallbacks, Writer

SAMPLE = sys.argv[1]
rows = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
font = extract_font(SAMPLE, rows, debug_path="out_debug.png")
print("xh", font.xh, "cap_h", font.cap_h, "stroke", round(font.stroke, 2))
print("synthesized:", "".join(add_fallbacks(font)))
font.save("profiles/trisha")

w = Writer(font, ink=(10, 90, 40), seed=3, neatness=0.5)
text = ("Suppose R3 has learned the full network graph.\n"
        "We know all the links, their status (up or down?), and their costs.\n"
        "R3 can run a graph algorithm to compute paths: next-hop = R2, cost = 10!")
w.write(text, width=1500, size=58, paper="lined").save("out_lined.png")
w.write(text, width=1500, size=58, paper="none").save("out_transparent.png")
