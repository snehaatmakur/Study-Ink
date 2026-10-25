import base64
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studyalong import grade, vision  # noqa: E402
from studyalong.grade import answer_shape, equivalent  # noqa: E402


# ----------------------------------------------------------------------------- helpers
class FakeClient:
    """Stands in for anthropic.Anthropic; records every request."""

    def __init__(self, transcription=None, same_meaning=False):
        self.calls = []
        self.transcription = transcription or {}
        self.same_meaning = same_meaning
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["system"].startswith(vision.TRANSCRIBE_SYSTEM):
            inp = {"text": "", "legible": True, "confidence": "high", "alternatives": [], **self.transcription}
        else:
            inp = {"same_meaning": self.same_meaning}
        # vision.py asks for a plain JSON text reply (no forced tool use)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(inp))])


def ink_png(text="R2", transparent=True):
    png = vision.synthetic_student_ink(text)
    if transparent:
        return png
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    out = io.BytesIO()
    Image.alpha_composite(bg, im).convert("RGB").save(out, "PNG")
    return out.getvalue()


def blank_png():
    buf = io.BytesIO()
    Image.new("RGBA", (200, 60), (0, 0, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


# ----------------------------------------------------------------------------- image prep
def test_blank_canvas_returns_none():
    assert vision.prepare_ink_image(blank_png()) is None


def test_prepare_flattens_transparent_ink_to_dark_on_white():
    p = vision.prepare_ink_image(ink_png("R2"))
    assert p is not None
    im = np.asarray(Image.open(io.BytesIO(p.png)).convert("L"))
    assert im[0, 0] == 255  # white margin, NOT black/transparent
    assert (im < 80).sum() > 50  # strokes are dark
    assert p.height >= 150  # upscaled to a readable size
    assert max(p.width, p.height) <= 1500


def test_prepare_handles_opaque_white_background_capture():
    p = vision.prepare_ink_image(ink_png("10", transparent=False))
    assert p is not None and (np.asarray(Image.open(io.BytesIO(p.png)).convert("L")) < 80).sum() > 30


# ----------------------------------------------------------------------------- matching
@pytest.mark.parametrize("read,answer,ok", [
    ("R2", "R2", True), ("r2", "R2", True), ("R 2", "R2", True), ("r-2", "R2", True),
    ("next hop R2", "R2", True), ("R3", "R2", False), ("R", "R2", False),
    ("5", "5", True), ("5.0", "5", True), ("S", "5", False), ("l0", "10", True), ("1O", "10", True),
    ("11", "10", False), ("5.0", "50", False), ("50", "5", False), ("10", "5", False),
    ("one source to all others", "one source to all others", True),
    ("One source to all others.", "one source to all others", True),
    ("one source to all other", "one source to all others", True),  # tiny slip
    ("one", "one source to all others", False),  # a lone word must NOT pass
    ("source", "one source to all others", False),
    ("", "R2", False),
    ("R3 → R2 → A", "R3 -> R2 -> A", True),
])
def test_equivalent(read, answer, ok):
    assert equivalent(read, answer) is ok


def test_answer_shape():
    assert answer_shape("10") == "number" and answer_shape("R2") == "label"
    assert answer_shape("A") == "label" and answer_shape("relaxation") == "phrase"
    assert answer_shape("one source to all others") == "phrase"


# ----------------------------------------------------------------------------- grading flow
def test_vision_correct_and_blind():
    fc = FakeClient({"text": "R2"})
    r = grade.grade_ink_png(ink_png("R2"), "R2", hint="h", client=fc)
    assert r["correct"] and r["method"] == "vision" and r["read"] == "R2" and r["answer"] is None
    # The model must never be told the answer key.
    texts = [c["text"] for m in fc.calls[0]["messages"] for c in m["content"] if c["type"] == "text"]
    assert all("R2" not in t for t in texts) and "R2" not in fc.calls[0]["system"]
    # And the image it received is the flattened, dark-on-white one.
    img = next(c for c in fc.calls[0]["messages"][0]["content"] if c["type"] == "image")
    raw = Image.open(io.BytesIO(base64.b64decode(img["source"]["data"]))).convert("L")
    assert raw.getpixel((0, 0)) == 255


def test_vision_wrong_gives_hint_not_answer():
    fc = FakeClient({"text": "R3", "confidence": "medium"})
    r = grade.grade_ink_png(ink_png("R3"), "R2", hint="Follow the cheapest path", client=fc)
    assert not r["correct"] and "Follow the cheapest path" in r["hint"] and r["answer"] is None
    assert r["read"] == "R3"


def test_model_cannot_self_grade():
    # Even if the model "agrees", only the code decides for a label.
    fc = FakeClient({"text": "R9", "alternatives": ["R2"]})
    assert not grade.grade_ink_png(ink_png("R9"), "R2", client=fc)["correct"]


def test_empty_canvas_makes_no_api_call():
    fc = FakeClient()
    r = grade.grade_ink_png(blank_png(), "R2", client=fc)
    assert r["empty"] and not r["correct"] and fc.calls == []


def test_unreadable_asks_for_typed_fallback():
    fc = FakeClient({"text": "", "legible": False, "confidence": "low"})
    r = grade.grade_ink_png(ink_png("R2"), "R2", client=fc)
    assert r["unreadable"] and r["needs_read"] and not r["correct"]


def test_no_key_falls_back_to_typed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    r = grade.grade_ink_png(ink_png("R2"), "R2")
    assert r["needs_read"] and r["method"] == "needs_read"


def test_api_error_is_reported_and_falls_back(monkeypatch):
    def boom(*a, **k):
        raise vision.VisionError("Anthropic rejected the API key (401).", "auth")
    monkeypatch.setattr(vision, "transcribe", boom)
    r = grade.grade_ink_png(ink_png("R2"), "R2")
    assert r["method"] == "vision_error" and r["needs_read"] and "401" in r["error"]


def test_typed_fallback_skips_vision():
    fc = FakeClient()
    r = grade.grade_ink_png(ink_png("R2"), "R2", client_read="r2", client=fc)
    assert r["correct"] and r["method"] == "client_read" and fc.calls == []


def test_phrase_semantic_fallback():
    fc = FakeClient({"text": "finds shortest paths from a single start node"}, same_meaning=True)
    r = grade.grade_ink_png(ink_png("x"), "one source to all others", client=fc)
    assert r["correct"] and r["method"] == "vision+semantic"
    fc2 = FakeClient({"text": "one"}, same_meaning=False)
    assert not grade.grade_ink_png(ink_png("x"), "one source to all others", client=fc2)["correct"]


def test_semantic_judge_not_used_for_short_answers():
    fc = FakeClient({"text": "R9"}, same_meaning=True)
    assert not grade.grade_ink_png(ink_png("R9"), "R2", client=fc)["correct"]
    assert len(fc.calls) == 1  # transcription only


# ----------------------------------------------------------------------------- HTTP server
@pytest.fixture
def server():
    import threading
    from http.server import ThreadingHTTPServer

    import app as studyink

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), studyink.H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _req(base, path, body=None):
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_server_endpoints(server):
    status, _, raw = _req(server, "/study/health")
    h = json.loads(raw)
    assert status == 200 and h["vision"]["enabled"] is True and h["vision"]["model"]

    status, _, raw = _req(server, "/handwriting/profiles")
    assert "trisha_full" in [p["name"] for p in json.loads(raw)["profiles"]]

    # text -> ink
    status, ctype, raw = _req(server, "/render", {"text": "Hello R2, cost = 5", "paper": "lined", "width": 1400})
    assert status == 200 and ctype == "image/png"
    assert Image.open(io.BytesIO(raw)).size[0] == 1400

    # Fill in the Ink (Study Companion): lessons, render, and an empty canvas through the real endpoint
    status, _, raw = _req(server, "/study/lessons")
    assert json.loads(raw)["lessons"]
    status, _, raw = _req(server, "/study/render", {"lesson_id": "dijkstra-r3"})
    assert status == 200 and json.loads(raw)["blanks"]
    b64 = base64.b64encode(blank_png()).decode()
    status, _, raw = _req(server, "/study/grade", {"blank_id": "b", "answer": "R2", "image_b64": b64, "kind": "table_hop"})
    assert status == 200 and json.loads(raw)["empty"] is True

    # typed fallback grades in code
    status, _, raw = _req(server, "/study/grade", {"blank_id": "b", "answer": "R2", "client_read": "r2"})
    assert json.loads(raw)["correct"] is True
