"""Export a rendered lesson as PDF + JSON for later Notability import."""
from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

from .page import RenderedLesson


def to_pdf(rendered: RenderedLesson) -> bytes:
    img = rendered.image.convert("RGB")
    if fitz is None:
        # Pillow PDF fallback
        buf = io.BytesIO()
        img.save(buf, "PDF", resolution=120.0)
        return buf.getvalue()
    doc = fitz.open()
    # letter-ish page sized to image aspect
    w, h = img.size
    page = doc.new_page(width=w, height=h)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def to_bundle(rendered: RenderedLesson) -> dict[str, Any]:
    return {
        "document": rendered.export_json(),
        "pdf_ready": True,
        "png_size": {"width": rendered.width, "height": rendered.height},
    }


def save_bundle(rendered: RenderedLesson, folder: str) -> dict[str, str]:
    import os

    os.makedirs(folder, exist_ok=True)
    png = os.path.join(folder, "page.png")
    pdf = os.path.join(folder, "page.pdf")
    meta = os.path.join(folder, "document.json")
    rendered.image.save(png)
    with open(pdf, "wb") as f:
        f.write(to_pdf(rendered))
    with open(meta, "w") as f:
        json.dump(rendered.export_json(), f, indent=2)
    return {"png": png, "pdf": pdf, "json": meta}
