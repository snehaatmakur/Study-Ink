"""Self-serve onboarding: hand the student a template, take back their filled-in sheet, and build a profile.

Two sheet kinds are offered:
  * "full"    - letters x2, capitals x2, digits x2, punctuation (the template `cli.py template` makes)
  * "letters" - one row of a-z and one row of A-Z; digits and punctuation fall back to stand-in glyphs

The row layout is derived from the kind alone, so the server never has to trust a client-supplied layout.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from .extract import ExtractionError, extract_font, ink_alpha, suspicious_variants
from .fallback import add_fallbacks
from .font import HandwritingFont
from .template import LOWER, UPPER, make_template, template_rows

SHEET_KINDS = ("full", "letters")
MARKER = "user_upload.json"   # present in profiles created through the upload flow (safe to replace)
MAX_SHEET_WIDTH = 2400        # photos are shrunk to this; the extractor is pixel-based and does not need more
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog.\nR3 to A: cost 3 (next-hop is R2)"


class SheetError(ValueError):
    """The uploaded sheet can't be turned into a profile; the message is written for the student."""


def rows_for(kind: str) -> list[str]:
    if kind == "full":
        return template_rows(2)
    if kind == "letters":
        return [LOWER, UPPER]
    raise SheetError("Unknown template type")


# ---- template files ---------------------------------------------------------------------------------

def template_png_bytes(kind: str = "full") -> bytes:
    rows = rows_for(kind)
    with tempfile.TemporaryDirectory() as tmp:
        png, js = os.path.join(tmp, "t.png"), os.path.join(tmp, "t.json")
        make_template(png, js, rows=rows)
        with open(png, "rb") as f:
            return f.read()


def template_pdf_bytes(kind: str = "full") -> bytes:
    """US Letter landscape at 300 dpi with the sheet scaled to fit inside 0.4in margins."""
    sheet = Image.open(io.BytesIO(template_png_bytes(kind))).convert("RGB")
    page_w, page_h, margin = 3300, 2550, 120
    scale = min((page_w - 2 * margin) / sheet.width, (page_h - 2 * margin) / sheet.height)
    sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)), Image.LANCZOS)
    page = Image.new("RGB", (page_w, page_h), "white")
    page.paste(sheet, (margin, margin))
    out = io.BytesIO()
    page.save(out, "PDF", resolution=300.0)
    return out.getvalue()


# ---- photo / scan cleanup ---------------------------------------------------------------------------

def _estimate_skew(gray_ink: np.ndarray, max_deg: float = 6.0, step: float = 0.25) -> float:
    """Angle (degrees) that best lines the rows of writing up horizontally.

    Rows of writing are separated by gaps, so the correct angle maximises the variance of the row-sum profile."""
    small = gray_ink
    if small.shape[1] > 800:
        f = 800 / small.shape[1]
        small = cv2.resize(small, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_score = 0.0, -1.0
    for ang in np.arange(-max_deg, max_deg + step, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = float(ang), score
    return best


def prepare_sheet(img: Image.Image) -> Image.Image:
    """Make a phone photo or scan look like the clean export the extractor was built for."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width > MAX_SHEET_WIDTH:
        img = img.resize((MAX_SHEET_WIDTH, round(img.height * MAX_SHEET_WIDTH / img.width)), Image.LANCZOS)
    arr = np.asarray(img)

    # 1. flatten uneven lighting: estimate the paper colour per channel (closing removes thin dark ink),
    #    then divide it out so shadows and a grey cast become white.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    paper = np.stack([cv2.GaussianBlur(cv2.morphologyEx(arr[..., c], cv2.MORPH_CLOSE, k), (0, 0), 15)
                      for c in range(3)], axis=2).astype(np.float32)
    flat = np.clip(arr.astype(np.float32) / np.maximum(paper, 1.0) * 255.0, 0, 255).astype(np.uint8)

    # 2. straighten a tilted page
    ink = (ink_alpha(Image.fromarray(flat)) * 255).astype(np.uint8)
    ang = _estimate_skew(ink)
    if abs(ang) >= 0.3:
        h, w = flat.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        flat = cv2.warpAffine(flat, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(flat)


# ---- profile building -------------------------------------------------------------------------------

def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:40]
    if not slug:
        raise SheetError("Give your handwriting a name (letters or numbers).")
    return slug


def friendly_extraction_error(exc: ExtractionError, kind: str) -> SheetError:
    tips = ("Tips: fill in every box, keep letters inside their own box and not touching their neighbours, "
            "write on the printed sheet only (no extra marks), and photograph it flat in good light.")
    return SheetError(f"{exc} {tips}")


def build_profile(image_bytes: bytes, name: str, kind: str, profiles_dir: str) -> dict:
    """Extract glyphs from the sheet and save profiles/<slug>. Returns info for the UI."""
    slug = slugify(name)
    rows = rows_for(kind)
    target = os.path.join(profiles_dir, slug)
    if os.path.isdir(target) and not os.path.isfile(os.path.join(target, MARKER)):
        raise SheetError(f"“{slug}” is already a built-in profile. Pick a different name.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        raise SheetError("That file couldn't be read as an image. Upload a PNG or JPEG of the filled-in sheet.") from exc

    sheet = prepare_sheet(img)
    if float((ink_alpha(sheet) > 0.5).mean()) < 0.0008:
        raise SheetError("I can't see any handwriting on that image. Did you upload the blank template? "
                         "Fill it in first, then upload it.")
    try:
        font = extract_font(sheet, rows)
    except ExtractionError as exc:
        raise friendly_extraction_error(exc, kind) from exc

    flagged = suspicious_variants(font)
    stand_ins = add_fallbacks(font)

    tmp = tempfile.mkdtemp(prefix=".upload-", dir=os.path.dirname(profiles_dir))
    try:
        font.save(tmp)
        with open(os.path.join(tmp, MARKER), "w") as f:
            json.dump({"kind": kind}, f)
        os.makedirs(profiles_dir, exist_ok=True)
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.replace(tmp, target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {"profile": slug, "kind": kind, "stand_ins": "".join(stand_ins), "flagged": flagged}


def preview_png_bytes(profile_dir: str) -> bytes:
    from .render import Writer
    font = HandwritingFont.load(profile_dir)
    img = Writer(font, ink=(10, 90, 40), seed=7, neatness=0.5).write(SAMPLE_TEXT, width=1100, size=46, paper="lined")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()
