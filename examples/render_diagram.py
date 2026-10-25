"""Render a small handwritten flowchart using a saved handwriting profile."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handwriting import DiagramWriter, HandwritingFont


font = HandwritingFont.load("profiles/trisha_full")
diagram = {
    "nodes": [
        {"id": "start", "kind": "box", "text": "Start", "x": 0.18, "y": 0.5, "w": 0.2, "h": 0.17},
        {"id": "ready", "kind": "diamond", "text": "Ready?", "x": 0.5, "y": 0.5, "w": 0.24, "h": 0.2},
        {"id": "done", "kind": "circle", "text": "Done", "x": 0.82, "y": 0.5, "w": 0.19, "h": 0.17},
    ],
    "edges": [
        {"from": "start", "to": "ready", "directed": True},
        {"from": "ready", "to": "done", "label": "yes", "directed": True},
    ],
}

DiagramWriter(font, ink=(25, 25, 30), seed=7, neatness=0.5).render(
    diagram, width=1200, height=600, label_size=42, paper="white"
).save("examples/diagram_sample.png")
