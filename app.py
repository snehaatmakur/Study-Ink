"""Study Ink: diagram → ink, text → ink, and Fill in the Ink (Study Companion).

Run:  python app.py   then open http://localhost:8000
"""
import base64
import io
import json
import os
import random
import re
import tempfile
import urllib.error
import urllib.request
from html import escape
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from studyalong.env import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY (and optional STUDYINK_VISION_MODEL) from .env

from handwriting import DiagramWriter, HandwritingFont, Writer, add_fallbacks
from handwriting import onboarding
from extractor import extract_text_from_file
from studyalong import vision
from studyalong.export import to_pdf
from studyalong.grade import grade_ink_png, grade_text
from studyalong.page import list_lessons, load_lesson, render_lesson, render_redraw_reference
from studyalong.solver import answers_for_lesson, forwarding_table, shortest_path

BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
APP_HTML = BASE_DIR / "app.html"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 28 * 1024 * 1024
_cache = {}

DIAGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "study_guide": {"type": "string"},
        "strokes": {"type": "array", "items": {"type": "object", "properties": {
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "kind": {"type": "string"},
        }, "required": ["points", "kind"], "additionalProperties": False}},
        "labels": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "text": {"type": "string"},
            "x": {"type": "number"}, "y": {"type": "number"},
            "w": {"type": "number"}, "h": {"type": "number"},
        }, "required": ["id", "text", "x", "y", "w", "h"], "additionalProperties": False}},
    },
    "required": ["title", "notes", "study_guide", "strokes", "labels"],
    "additionalProperties": False,
}

VISION_SYSTEM = """You convert the central technical diagram in an image into a redrawable scene description.
Return one valid JSON object only, without markdown, with this shape:
{"title":"slide title","notes":["important lecture text"],"study_guide":"A source-grounded explanation of the slide.","strokes":[{"points":[[0.1,0.2],[0.3,0.4]],"kind":"single"}],"labels":[{"id":"l1","text":"N","x":0.5,"y":0.5,"w":0.03,"h":0.04}]}
Coordinates are normalized 0..1 over the full image. Put the slide title in title, important lecture prose and bullets in notes in reading order, and a concise explanation based only on the slide in study_guide. Keep lecture prose out of diagram labels. By default, describe the entire diagram geometry as connected polylines in strokes (include molecular rings, single/double bonds, arrows, axes, and outlines), and every visible diagram text/atom symbol as labels with tight normalized bounding boxes. For chemistry, each atom token is a separate label, and each bond segment is a stroke; use two nearby parallel strokes for double bonds. Preserve visible charges, subscripts, capitalization and punctuation. If the user enables adaptive simplification, instead make a legible explanatory schematic when the source is too dense: retain the main components, important labels, and primary relationships, and abstract secondary detail. Do not invent connections. Do not return an empty result if uncertain: include the visible parts and leave unreadable text out."""


def get_font(name):
    if name not in _cache:
        if name not in {p.name for p in PROFILES_DIR.iterdir() if (p / "meta.json").is_file()}:
            raise ValueError("Unknown handwriting profile")
        font = HandwritingFont.load(str(PROFILES_DIR / name))
        add_fallbacks(font)  # stand-in glyphs for characters the student has not written yet
        _cache[name] = font
    return _cache[name]


def list_profiles():
    """Every profile folder, flagged when it was created through the in-app upload flow."""
    if not PROFILES_DIR.is_dir():
        return []
    return [{"name": p.name, "custom": (p / onboarding.MARKER).is_file()}
            for p in sorted(PROFILES_DIR.iterdir()) if (p / "meta.json").is_file()]


def hex_to_rgb(value):
    value = value.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError("Ink color must be a six-digit hex color")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


STOP_WORDS = set("about above after again against all also always and any are around because been before being below between both but could did does doing down during each from further had has have having her here hers herself him himself his how into its itself just know like more most much myself never nor not now once only other our ours ourselves out over own really same she should some such than that the their theirs them themselves then there these they this those through too under until very was were what when where which while who whom why will with would you your yours yourself yourselves suppose think".split())


def apply_active_recall(text, seed=7):
    tokens = re.split(r"(\W+)", text)
    candidates = {}
    for index, token in enumerate(tokens):
        if token.isalpha() and len(token) > 3 and token.lower() not in STOP_WORDS:
            candidates[index] = 2 if token[0].isupper() else 1
    if not candidates:
        return text
    count = max(1, round(len(candidates) * 0.25))
    rng = random.Random(int(seed))
    selected = set()
    remaining = dict(candidates)
    while len(selected) < count:
        index = rng.choices(list(remaining), weights=list(remaining.values()), k=1)[0]
        selected.add(index)
        remaining.pop(index)
    for index in selected:
        tokens[index] = "_" * len(tokens[index])
    return "".join(tokens)


def parse_model_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Claude did not return a JSON diagram. Try analyzing again or adjust the prompt.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Claude returned incomplete or invalid diagram data. Try a tighter crop and analyze again.") from exc
    if not isinstance(data, dict):
        raise ValueError("Claude returned an invalid diagram object")
    labels = data.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("Diagram JSON must have a labels array")

    clean_labels = []
    for i, original in enumerate(labels):
        if not isinstance(original, dict):
            continue
        node = dict(original)
        node["id"] = str(node.get("id") or f"l{i + 1}")
        node["text"] = str(node.get("text") or "")
        for key, default in (("x", 0.5), ("y", 0.5), ("w", 0.14), ("h", 0.1)):
            try:
                node[key] = float(node.get(key, default))
            except (TypeError, ValueError):
                node[key] = default
        node["x"] = min(1.0, max(0.0, node["x"]))
        node["y"] = min(1.0, max(0.0, node["y"]))
        node["w"] = min(1.0, max(0.01, node["w"]))
        node["h"] = min(1.0, max(0.01, node["h"]))
        clean_labels.append(node)
    strokes = []
    for item in data.get("strokes", []):
        if not isinstance(item, dict) or not isinstance(item.get("points"), list):
            continue
        points = []
        for point in item["points"]:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    points.append([min(1.0, max(0.0, float(point[0]))), min(1.0, max(0.0, float(point[1])))])
                except (TypeError, ValueError):
                    continue
        if len(points) >= 2:
            strokes.append({"points": points, "kind": str(item.get("kind") or "single")})
    notes = data.get("notes", [])
    if isinstance(notes, str):
        notes = notes.splitlines()
    if not isinstance(notes, list):
        notes = []
    return {
        "title": str(data.get("title") or ""),
        "notes": [str(note).strip() for note in notes if str(note).strip()],
        "study_guide": str(data.get("study_guide") or "").strip(),
        "labels": clean_labels,
        "strokes": strokes,
    }


def write_a4_pages(writer, text, width=1240, height=1754, size=40, paper="lined"):
    """Splits text into A4 page chunks with proper line height margins."""
    lines = text.splitlines()

    # Calculate line capacity per page based on font size
    line_height = max(30, int(size * 1.5))
    top_bottom_margin = 160
    usable_height = height - top_bottom_margin
    max_lines_per_page = max(1, usable_height // line_height)

    # Wrap long lines so they don't overflow the page vertically
    wrapped_lines = []
    max_chars = max(20, int(width / (size * 0.55)))
    for line in lines:
        if not line.strip():
            wrapped_lines.append("")
            continue
        while len(line) > max_chars:
            # Wrap at nearest space if possible
            space_idx = line.rfind(" ", 0, max_chars)
            if space_idx > 0:
                wrapped_lines.append(line[:space_idx])
                line = line[space_idx + 1 :]
            else:
                wrapped_lines.append(line[:max_chars])
                line = line[max_chars:]
        wrapped_lines.append(line)

    pages = []
    current_chunk = []

    for line in wrapped_lines:
        current_chunk.append(line)
        if len(current_chunk) >= max_lines_per_page:
            img = writer.write(
                "\n".join(current_chunk),
                width=width,
                size=size,
                paper=paper,
            )
            pages.append(img)
            current_chunk = []

    if current_chunk or not pages:
        img = writer.write(
            "\n".join(current_chunk or [""]),
            width=width,
            size=size,
            paper=paper,
        )
        pages.append(img)

    return pages

def _ssl_context():
    """Trust certifi's CA bundle when available (python.org Python on macOS ships without root certs)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def analyze_slide(image_bytes, media_type, model, extra_instructions="", current_diagram=None,
                  simplify_complex=False):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Put it in the .env file next to app.py (see .env.example) and restart the app.")
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("Choose a PNG, JPEG, WebP, or GIF slide image")
    if current_diagram:
        preservation = ("preserve the title, lecture notes, study guide, main components, important labels, primary relationships, and overall "
                        "layout; you may consolidate dense secondary detail if adaptive simplification is enabled. "
                        if simplify_complex else
                        "preserve every existing label, line, bond, position, and relationship unless the user explicitly asks to change it. ")
        prompt = ("Revise the current diagram description according to the user's requested edit. "
                  "Use the source image as the ground truth: " + preservation + "Return the full "
                  "revised diagram in the required JSON schema, not a patch.\n\nCurrent diagram JSON:\n" +
                  json.dumps(current_diagram, ensure_ascii=False)[:30000])
    else:
        prompt = ("Read the slide and return its title, key lecture notes in reading order, and a concise, source-grounded study guide. "
                  "Also create a legible explanatory schematic of the central diagram: retain its main components, "
                  "important labels, and primary connections; abstract secondary detail only if the source is "
                  "too dense to redraw clearly. For chemistry, preserve named bases and pairing/backbone relationships."
                  if simplify_complex else
                  "Read the slide title, important lecture text and bullets in reading order, and write a concise, source-grounded study guide. "
                  "Then redraw the central diagram: return all visible bond/line geometry as normalized polylines and all diagram text as normalized label boxes, following the required JSON schema. For chemistry include every atom token and every bond.")
    if extra_instructions.strip():
        prompt += "\n\nAdditional user guidance for this slide:\n" + extra_instructions.strip()[:4000]
    if simplify_complex:
        prompt += ("\n\nADAPTIVE SIMPLIFICATION IS ENABLED. First identify the diagram's main idea, essential "
                  "entities, labels, and relationships. If it is crowded, unusually dense, or contains "
                  "more detail than can be represented legibly, return a simplified explanatory schematic "
                  "instead of trying to copy every mark. Keep the main components, their relative layout, "
                  "all important labels, and the primary connections/arrows. Remove decorative marks, "
                  "redundant detail, and secondary substructure. For chemistry, preserve the named bases "
                  "and pairing/backbone relationships; dense atom-by-atom rings may be represented by clearly "
                  "labeled base shapes. Do not invent facts or connections. If simplified, prefix the title "
                  "with 'Simplified — '. If the source is already clear and manageable, keep it faithful.")
    body = {
        "model": model or vision.vision_model(),
        "max_tokens": 8192,
        "system": VISION_SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": DIAGRAM_SCHEMA}},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                               "data": base64.b64encode(image_bytes).decode("ascii")}},
            {"type": "text", "text": prompt},
        ]}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120, context=_ssl_context()) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise RuntimeError(f"Claude API error ({exc.code}): {detail or 'check your API key, model, and account access'}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        hint = (" Python can't verify SSL certificates: run `pip install certifi` (on macOS python.org installs, also run "
            "'Install Certificates.command' in /Applications/Python 3.x).") if "CERTIFICATE" in str(reason).upper() else ""
        raise RuntimeError(f"Could not reach the Claude API ({reason}).{hint}") from exc
    text = "".join(block.get("text", "") for block in answer.get("content", []) if block.get("type") == "text")
    if answer.get("stop_reason") == "max_tokens":
        raise RuntimeError("Claude's diagram description was too large and got cut off. Crop closer to the diagram and analyze again.")
    return parse_model_json(text)


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def render_study_lesson(data):
    lesson_id = str(data.get("lesson_id", "dijkstra-r3"))
    if not re.fullmatch(r"[a-z0-9-]{1,80}", lesson_id):
        raise ValueError("Invalid lesson id")
    return render_lesson(
        lesson_id,
        profile=str(data.get("profile", "profiles/trisha_full")),
        ink=hex_to_rgb(str(data.get("ink", "#0c582a"))),
        size=min(120.0, max(16.0, float(data.get("size", 40)))),
        width=min(2200, max(500, int(data.get("width", 1100)))),
        paper=data.get("paper", "lined"),
        seed=int(data.get("seed", 7)),
        neatness=min(1.0, max(0.0, float(data.get("neatness", 0.55)))),
        reveal=set(data.get("reveal", [])),
        mode=data.get("mode", "study"),
    )


class H(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > MAX_REQUEST_BYTES:
            raise ValueError("Request is too large")
        return json.loads(self.rfile.read(size))

    def send_bytes(self, content_type, body, filename=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        split = urlsplit(self.path)
        path = split.path
        if path in {"/handwriting/template.png", "/handwriting/template.pdf"}:
            kind = parse_qs(split.query).get("kind", ["full"])[0]
            try:
                if path.endswith(".png"):
                    self.send_bytes("image/png", onboarding.template_png_bytes(kind), f"study-ink-template-{kind}.png")
                else:
                    self.send_bytes("application/pdf", onboarding.template_pdf_bytes(kind), f"study-ink-template-{kind}.pdf")
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if path == "/handwriting/profiles":
            self.send_json(200, {"profiles": list_profiles()})
            return
        if path == "/study/health":
            self.send_json(200, {"ok": True, "service": "study-ink",
                                 "vision": {"enabled": vision.vision_enabled(), "model": vision.vision_model()}})
            return
        if path == "/study/vision/selftest":
            # Round-trips synthetic handwriting through the real grading path (~3 small API calls).
            self.send_json(200, vision.selftest())
            return
        if path == "/study/lessons":
            self.send_json(200, {"lessons": list_lessons()})
            return
        match = re.fullmatch(r"/study/lessons/([a-z0-9-]{1,80})", path)
        if match:
            try:
                lesson = load_lesson(match.group(1))
            except FileNotFoundError:
                self.send_json(404, {"error": "Lesson not found"})
                return
            public = {**lesson, "blank_ids": list(answers_for_lesson(lesson))}
            self.send_json(200, public)
            return
        match = re.fullmatch(r"/study/redraw/([a-z0-9-]{1,80})/([a-zA-Z0-9_-]{1,80})", path)
        if match:
            try:
                image = render_redraw_reference(match.group(1), match.group(2))
            except (FileNotFoundError, StopIteration):
                self.send_json(404, {"error": "Redraw reference not found"})
                return
            output = png_bytes(image)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
            return
        if path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        profiles = [p["name"] for p in list_profiles()]
        options = "".join(f'<option value="{escape(p)}"{" selected" if p == "trisha_full" else ""}>{escape(p)}</option>' for p in profiles)
        page = APP_HTML.read_text(encoding="utf-8").replace("__PROFILES__", options).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        try:
            data = self.read_json()
            if self.path == "/handwriting/upload":
                image_bytes = base64.b64decode(data.get("image_base64", ""), validate=True)
                if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("The sheet image must be under 10 MB")
                kind = str(data.get("kind", "full"))
                result = onboarding.build_profile(image_bytes, str(data.get("name", "")), kind, str(PROFILES_DIR))
                _cache.pop(result["profile"], None)
                result["preview_b64"] = base64.b64encode(
                    onboarding.preview_png_bytes(str(PROFILES_DIR / result["profile"]))).decode("ascii")
                result["profiles"] = list_profiles()
                self.send_json(200, result)
                return
            if self.path == "/study/render":
                rendered = render_study_lesson(data)
                self.send_json(200, {
                    "width": rendered.width,
                    "height": rendered.height,
                    "blanks": rendered.blanks,
                    "image_b64": base64.b64encode(rendered.png_bytes()).decode("ascii"),
                    "document": rendered.export_json(),
                })
                return
            if self.path == "/study/grade":
                answer = str(data.get("answer", ""))
                if data.get("image_b64"):
                    ink = base64.b64decode(data["image_b64"], validate=True)
                    if len(ink) > MAX_IMAGE_BYTES:
                        raise ValueError("The ink sample is too large")
                    result = grade_ink_png(ink, answer, str(data.get("hint", "")),
                                           client_read=data.get("client_read") or None,
                                           kind=str(data.get("kind", "")))
                elif data.get("client_read") is not None:
                    result = grade_text(str(data["client_read"]), answer, str(data.get("hint", "")))
                else:
                    raise ValueError("Draw an answer or type what you wrote first")
                self.send_json(200, result)
                return
            if self.path == "/study/export/pdf":
                rendered = render_study_lesson(data)
                output = to_pdf(rendered)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", "attachment; filename=study-along.pdf")
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
                return
            if self.path == "/study/solve":
                lesson_id = str(data.get("lesson_id", "dijkstra-r3"))
                if not re.fullmatch(r"[a-z0-9-]{1,80}", lesson_id):
                    raise ValueError("Invalid lesson id")
                lesson = load_lesson(lesson_id)
                graphs = {block["id"]: block for block in lesson.get("blocks", []) if block.get("type") == "graph"}
                graph = graphs.get(str(data.get("graph", "")))
                if not graph:
                    raise ValueError("Graph not found in lesson")
                source = str(data.get("source", graph.get("source", "")))
                target = data.get("target")
                result = {"table": forwarding_table(graph, source)}
                if target:
                    result["shortest"] = shortest_path(graph, source, str(target))
                self.send_json(200, result)
                return
            if self.path == "/analyze-slide":
                image_bytes = base64.b64decode(data.get("image_base64", ""), validate=True)
                if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("Slide image must be under 10 MB")
                diagram = analyze_slide(image_bytes, data.get("media_type", ""),
                                       data.get("model") or vision.vision_model(),
                                       str(data.get("extra_instructions", "")),
                                       data.get("current_diagram"),
                                       simplify_complex=bool(data.get("simplify_complex", False)))
                self.send_json(200, {"diagram": diagram})
                return

            if self.path == "/extract":
                filename = Path(str(data.get("filename", "upload"))).name
                suffix = Path(filename).suffix.lower()
                if suffix not in {".txt", ".pdf", ".png", ".jpg", ".jpeg"}:
                    raise ValueError("Choose a TXT, PDF, PNG, or JPEG file")
                file_bytes = base64.b64decode(data.get("data", ""), validate=True)
                if not file_bytes or len(file_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("File must be under 10 MB")
                with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                    temp.write(file_bytes)
                    temp.flush()
                    extracted = extract_text_from_file(temp.name)
                if extracted.startswith("Error:"):
                    raise ValueError(extracted)
                self.send_json(200, {"text": extracted, "filename": filename})
                return

            if self.path == "/render-over-source":
                source_bytes = base64.b64decode(data.get("source_base64", ""), validate=True)
                if not source_bytes or len(source_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("Upload a slide image under 10 MB first")
                source = Image.open(io.BytesIO(source_bytes))
                source.load()
                labels = (data.get("diagram") or {}).get("labels") or []
                writer = DiagramWriter(get_font(data.get("profile", "trisha_full")),
                                       seed=int(data.get("seed", 7)),
                                       neatness=min(1.0, max(0.0, float(data.get("neatness", 0.5)))))
                image = writer.render_over_source(source, labels,
                                                  label_size=min(80, max(16, float(data.get("label_size", 42)))))
                output = png_bytes(image)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(output)))
                self.send_header("X-Labels-Kept-Printed", str(len(writer.unrendered_labels)))
                self.end_headers()
                self.wfile.write(output)
                return

            if self.path == "/render-redrawn":
                source_bytes = base64.b64decode(data.get("source_base64", ""), validate=True)
                if not source_bytes or len(source_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("Upload a slide image under 10 MB first")
                source = Image.open(io.BytesIO(source_bytes))
                source.load()
                writer = DiagramWriter(get_font(data.get("profile", "trisha_full")), seed=int(data.get("seed", 7)),
                                       neatness=min(1.0, max(0.0, float(data.get("neatness", 0.5)))))
                image = writer.render_redrawn(source, data.get("diagram") or {},
                                              label_size=min(80, max(16, float(data.get("label_size", 42)))))
                output = png_bytes(image)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(output)))
                self.send_header("X-Labels-Kept-Printed", str(len(writer.unrendered_labels)))
                self.end_headers()
                self.wfile.write(output)
                return

            if self.path == "/render-diagram":
                diagram = data.get("diagram") or {}
                image = DiagramWriter(get_font(data.get("profile", "trisha_full")),
                                      seed=int(data.get("seed", 7)),
                                      neatness=min(1.0, max(0.0, float(data.get("neatness", 0.5))))).render(
                    diagram, width=1200, height=800,
                    label_size=min(80, max(16, float(data.get("label_size", 42)))),
                    paper=data.get("paper", "white"))
                output = png_bytes(image)
            elif self.path == "/render":
                render_text = str(data.get("text", ""))
                if data.get("active_recall"):
                    render_text = apply_active_recall(
                        render_text, data.get("seed", 7)
                    )

                profile_name = data.get("profile", "trisha_full")
                ink_color = hex_to_rgb(data.get("ink", "#0a5a28"))
                seed_val = int(data.get("seed", 7))
                neatness_val = float(data.get("neatness", 0.5))

                writer = Writer(
                    get_font(profile_name),
                    ink=ink_color,
                    seed=seed_val,
                    neatness=neatness_val,
                )

                # Render continuous single image matching requested container width
                page_width = int(data.get("width", 1400))
                page_size = float(data.get("size", 40))
                paper_style = data.get("paper", "lined")

                image = writer.write(
                    render_text,
                    width=page_width,
                    size=page_size,
                    paper=paper_style,
                )
                output = png_bytes(image)

                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(503, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *_):
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Study Ink: open http://localhost:{port}  (Ctrl+C to stop)")
    print("Claude vision:", f"on ({vision.vision_model()})" if vision.vision_enabled() else "off (no ANTHROPIC_API_KEY)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()