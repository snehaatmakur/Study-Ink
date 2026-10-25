"""Render simple node-and-edge diagrams with a student's handwriting glyphs."""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .font import HandwritingFont
from .render import CHAR_MAP, SCRIPT_CHARS, Writer


class DiagramWriter:
    """Draw simple diagrams and label them with the same glyphs used by ``Writer``.

    Node positions and sizes are normalized to the canvas (0..1). A node's x/y
    is its center; w/h are its full width and height. Edges refer to node ids.
    """

    def __init__(self, font: HandwritingFont, ink=(25, 25, 30), seed: int | None = 7,
                 neatness: float = 0.5):
        self.font = font
        self.ink = tuple(ink)
        self.seed = seed
        self.neatness = neatness
        self.rng = random.Random(seed)

    def render(self, diagram: dict[str, Any], width: int = 1200, height: int = 800,
               label_size: float = 34, paper: str = "white") -> Image.Image:
        """Render a diagram dict containing ``nodes`` and ``edges`` to a PIL image."""
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if paper not in {"white", "lined", "grid"}:
            raise ValueError("paper must be 'white', 'lined', or 'grid'")

        image = self._paper(width, height, paper)
        draw = ImageDraw.Draw(image)
        nodes = diagram.get("nodes") or []
        edges = diagram.get("edges") or []
        by_id = {str(n["id"]): n for n in nodes if "id" in n}
        boxes = {key: self._box(node, width, height) for key, node in by_id.items()}

        # Draw connectors first so node outlines and labels sit clearly above them.
        for edge in edges:
            a_id, b_id = str(edge.get("from", "")), str(edge.get("to", ""))
            if a_id not in boxes or b_id not in boxes:
                continue
            a, b = boxes[a_id], boxes[b_id]
            start = self._border_point(a, b[0], b[1])
            end = self._border_point(b, a[0], a[1])
            self._wobbly_line(draw, start, end, width)
            if edge.get("directed"):
                self._arrow(draw, start, end, width)
            label = str(edge.get("label") or "").strip()
            if label:
                mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                # A small paper-colored patch keeps connector labels legible.
                self._draw_label(image, label, mx, my - label_size * 0.65, label_size * 0.72,
                                 max_width=max(90, abs(end[0] - start[0]) * 0.8))

        for key, node in by_id.items():
            x, y, w, h = boxes[key]
            kind = node.get("kind", "box")
            if kind != "label":
                self._shape(draw, kind, (x, y, w, h), width)
            text = str(node.get("text") or "").strip()
            if text:
                max_text_width = max(48, w - min(24, w * 0.12))
                size = min(label_size, max(16, h * 0.48)) if kind != "label" else label_size * 0.8
                self._draw_label(image, text, x, y, size, max_width=max_text_width)

        return image

    def render_over_source(self, source: Image.Image, labels: list[dict[str, Any]],
                           label_size: float = 42) -> Image.Image:
        """Replace printed labels with handwriting while retaining source geometry.

        This mode is useful for dense technical drawings where a model can read
        labels more reliably than it can reconstruct every bond or connector.
        Label x/y are normalized centers; w/h are normalized bounding-box size.
        """
        image = source.convert("RGB").copy()
        width, height = image.size
        draw = ImageDraw.Draw(image)
        self.unrendered_labels = []
        for label in labels:
            text = str(label.get("text") or "").strip()
            if not text:
                continue
            # Keep unsupported source text intact instead of silently deleting
            # a symbol for which this handwriting profile has no glyph.
            if not self._supports_text(text):
                self.unrendered_labels.append(text)
                continue
            cx = float(label.get("x", 0.5)) * width
            cy = float(label.get("y", 0.5)) * height
            box_w = max(1, float(label.get("w", 0.04)) * width)
            box_h = max(1, float(label.get("h", 0.04)) * height)
            pad = max(2, min(6, round(min(box_w, box_h) * 0.12)))
            x0 = max(0, round(cx - box_w / 2) - pad)
            y0 = max(0, round(cy - box_h / 2) - pad)
            x1 = min(width, round(cx + box_w / 2) + pad)
            y1 = min(height, round(cy + box_h / 2) + pad)

            # Match the local paper color instead of assuming a pure white slide.
            ring = image.crop((max(0, x0 - 3), max(0, y0 - 3), min(width, x1 + 3), min(height, y1 + 3)))
            pixels = np.asarray(ring).reshape(-1, 3)
            light = pixels[pixels.mean(axis=1) > 150]
            bg = tuple(np.median(light if len(light) else pixels, axis=0).astype(np.uint8))
            draw.rectangle((x0, y0, x1, y1), fill=bg)

            # Scale to the source label's height. Keep the entire label on one line.
            size = min(label_size, max(12, box_h * 1.65))
            self._draw_label(image, text, cx, cy, size, max_width=10000)
        return image

    def render_redrawn(self, source: Image.Image, diagram: dict[str, Any],
                       label_size: float = 42) -> Image.Image:
        """Redraw a slide's diagram from model-provided strokes and glyph labels."""
        width, height = source.size
        image = Image.new("RGB", (width, height), (250, 250, 247))
        draw = ImageDraw.Draw(image)
        self.unrendered_labels = []
        for stroke in diagram.get("strokes", []):
            points = stroke.get("points", [])
            if len(points) < 2:
                continue
            scaled = [(float(x) * width, float(y) * height) for x, y in points]
            for start, end in zip(scaled, scaled[1:]):
                self._wobbly_line(draw, start, end, width)
        for label in diagram.get("labels", []):
            text = str(label.get("text") or "").strip()
            if not text:
                continue
            if not self._supports_text(text):
                self.unrendered_labels.append(text)
                continue
            x, y = float(label.get("x", 0.5)) * width, float(label.get("y", 0.5)) * height
            h = float(label.get("h", 0.04)) * height
            self._draw_label(image, text, x, y, min(label_size, max(12, h * 1.65)), max_width=10000)
        return image

    def _supports_text(self, text: str) -> bool:
        for char in text:
            if char.isspace():
                continue
            mapped = CHAR_MAP.get(char, char)
            for normalized in mapped:
                script = SCRIPT_CHARS.get(normalized)
                glyph = script[0] if script else normalized
                if not self.font.has(glyph):
                    return False
        return True

    def _paper(self, width: int, height: int, paper: str) -> Image.Image:
        image = Image.new("RGB", (width, height), (250, 250, 247))
        draw = ImageDraw.Draw(image)
        color = (218, 225, 236)
        if paper == "lined":
            for y in range(42, height, 44):
                draw.line((0, y, width, y), fill=color, width=1)
        elif paper == "grid":
            for x in range(0, width, 36):
                draw.line((x, 0, x, height), fill=color, width=1)
            for y in range(0, height, 36):
                draw.line((0, y, width, y), fill=color, width=1)
        return image

    @staticmethod
    def _box(node: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
        x, y = float(node.get("x", 0.5)) * width, float(node.get("y", 0.5)) * height
        w, h = float(node.get("w", 0.14)) * width, float(node.get("h", 0.09)) * height
        if w <= 0 or h <= 0:
            raise ValueError(f"node {node.get('id', '?')} has non-positive size")
        return x, y, w, h

    @staticmethod
    def _border_point(box: tuple[float, float, float, float], tx: float, ty: float) -> tuple[float, float]:
        x, y, w, h = box
        dx, dy = tx - x, ty - y
        if abs(dx) + abs(dy) < 1e-6:
            return x, y
        scale = min((w / 2) / abs(dx) if dx else float("inf"),
                    (h / 2) / abs(dy) if dy else float("inf"))
        return x + dx * scale, y + dy * scale

    def _pen_width(self, canvas_width: int) -> int:
        return max(2, round(self.font.stroke * canvas_width / 1400 * 1.25))

    def _wobbly_line(self, draw: ImageDraw.ImageDraw, start: tuple[float, float],
                     end: tuple[float, float], canvas_width: int) -> None:
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        count = max(3, int(length / 32))
        wobble = min(2.2, max(0.6, canvas_width / 800)) * (1.1 - self.neatness * 0.65)
        points = []
        for i in range(count + 1):
            t = i / count
            offset = self.rng.uniform(-wobble, wobble) if 0 < i < count else 0
            points.append((x1 + (x2 - x1) * t + nx * offset,
                           y1 + (y2 - y1) * t + ny * offset))
        draw.line(points, fill=self.ink, width=self._pen_width(canvas_width), joint="curve")

    def _shape(self, draw: ImageDraw.ImageDraw, kind: str,
               box: tuple[float, float, float, float], canvas_width: int) -> None:
        x, y, w, h = box
        half_w, half_h = w / 2, h / 2
        if kind in {"circle", "ellipse"}:
            points = []
            steps = max(28, int((w + h) / 5))
            wobble = min(2.0, canvas_width / 900) * (1.1 - self.neatness * 0.65)
            for i in range(steps + 1):
                angle = 2 * math.pi * i / steps
                radius = self.rng.uniform(-wobble, wobble)
                points.append((x + (half_w + radius) * math.cos(angle),
                               y + (half_h + radius) * math.sin(angle)))
            draw.line(points, fill=self.ink, width=self._pen_width(canvas_width), joint="curve")
            return
        if kind == "diamond":
            corners = [(x, y - half_h), (x + half_w, y), (x, y + half_h), (x - half_w, y)]
        else:  # default box
            corners = [(x - half_w, y - half_h), (x + half_w, y - half_h),
                       (x + half_w, y + half_h), (x - half_w, y + half_h)]
        for i, start in enumerate(corners):
            self._wobbly_line(draw, start, corners[(i + 1) % len(corners)], canvas_width)

    def _arrow(self, draw: ImageDraw.ImageDraw, start: tuple[float, float],
               end: tuple[float, float], canvas_width: int) -> None:
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head = max(10, canvas_width * 0.012)
        for delta in (-math.pi / 6, math.pi / 6):
            tip = (end[0] - head * math.cos(angle + delta),
                   end[1] - head * math.sin(angle + delta))
            self._wobbly_line(draw, tip, end, canvas_width)

    def _draw_label(self, image: Image.Image, text: str, x: float, y: float,
                    size: float, max_width: float) -> None:
        # Delegate every character to the same profile-backed text writer.
        text_writer = Writer(self.font, ink=self.ink, seed=self.rng.randrange(2**31),
                             neatness=self.neatness)
        width = max(80, int(max_width))
        label = text_writer.write(text, width=width, size=size, margin=0, paper="none", align="center")
        alpha = label.getchannel("A")
        bounds = alpha.getbbox()
        if not bounds:
            return
        label = label.crop(bounds)
        paste_x = round(x - label.width / 2)
        paste_y = round(y - label.height / 2)
        # White label backing hides a connector beneath edge text.
        if max_width < 200:
            ImageDraw.Draw(image).rounded_rectangle(
                (paste_x - 5, paste_y - 3, paste_x + label.width + 5, paste_y + label.height + 3),
                radius=4, fill=(250, 250, 247))
        image.paste(label, (paste_x, paste_y), label)
