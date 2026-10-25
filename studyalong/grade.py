"""Grade a student's handwriting in a blank.

Flow for ink: prepare crop -> Claude transcribes it BLIND -> THIS FILE compares the transcription to
the answer key in code. The model never decides correctness for short answers (numbers, labels).
For longer free-text phrases that fail the strict check, a text-only paraphrase judge gets a say.

Without ANTHROPIC_API_KEY, or if the API fails, the result carries `needs_read: true` so the UI can
fall back to the student typing what they wrote.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from . import vision

FILLER_WORDS = {"next", "hop", "router", "node", "cost", "is", "the", "to", "via", "answer", "ans"}
# Handwriting confusions that are safe to forgive ONLY when the expected answer is numeric.
DIGIT_LOOKALIKES = str.maketrans({"o": "0", "l": "1", "i": "1", "|": "1"})


# --------------------------------------------------------------------------- matching
def answer_shape(answer: str) -> str:
    """number | label | phrase. Describes the *form* of the key; used to guide OCR and matching."""
    a = answer.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", a):
        return "number"
    if re.fullmatch(r"[A-Za-z]{1,2}\d{0,2}", a):
        return "label"
    return "phrase"


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("→", " ").replace("->", " ").replace("—", "-").replace("–", "-")
    s = re.sub(r"[^\w\s\-\+\=\.\,/:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _compact(s: str) -> str:
    return re.sub(r"[\s\-\.,]", "", s)


def equivalent(read: str, answer: str, shape: str | None = None) -> bool:
    """Strict, deterministic comparison of what was read against the answer key."""
    a, b = _norm(read), _norm(answer)
    if not a or not b:
        return False
    if a == b:
        return True
    shape = shape or answer_shape(answer)

    if shape == "number":
        nospace = re.sub(r"\s", "", a)
        for cand in (nospace, nospace.translate(DIGIT_LOOKALIKES)):
            try:
                return abs(float(cand) - float(re.sub(r"\s", "", b))) < 1e-6
            except ValueError:
                continue
        return False

    ca, cb = _compact(a), _compact(b)
    if ca == cb:
        return True

    if shape == "label":
        # "next hop R2", "R 2", "r-2"  ->  R2.  Only whole tokens count, and the writing must be short.
        tokens = [t for t in re.split(r"[\s,;:/]+", a) if t and t not in FILLER_WORDS]
        return len(tokens) == 1 and _compact(tokens[0]) == cb

    # phrase: tolerate tiny spelling slips, but never a lone word out of a longer phrase
    wa, wb = a.split(), b.split()
    if abs(len(wa) - len(wb)) > 1:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


# --------------------------------------------------------------------------- results
def _result(read: str, correct: bool, hint: str, method: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "read": read,
        "correct": correct,
        "hint": "" if correct else hint,
        "answer": None,  # never echoed back; the UI's Reveal button is the only way to see it
        "method": method,
        "needs_read": False,
        "vision_enabled": vision.vision_enabled(),
    }
    out.update(extra)
    return out


def _default_hint(hint: str) -> str:
    return hint or "Not quite — look at the slide again, then try once more."


def grade_text(read: str, answer: str, hint: str = "") -> dict[str, Any]:
    ok = equivalent(read, answer)
    return _result(read, ok, _default_hint(hint), "text")


def grade_ink_png(
    png_bytes: bytes,
    answer: str,
    hint: str = "",
    client_read: str | None = None,
    kind: str = "",
    client: Any | None = None,
) -> dict[str, Any]:
    """Read the handwriting in `png_bytes` with Claude and check it against `answer`."""
    # 1. The student typed what they wrote (fallback path) - grade that directly.
    if client_read is not None and client_read.strip():
        res = grade_text(client_read, answer, hint)
        res["method"] = "client_read"
        return res

    # 2. Nothing written? Don't spend an API call.
    try:
        prepared = vision.prepare_ink_image(png_bytes)
    except Exception as e:  # corrupt/unsupported image
        return _result("", False, f"I couldn't open that drawing ({type(e).__name__}). Clear it and try again.", "bad_image")
    if prepared is None:
        return _result("", False, "Nothing written yet — write your answer in the box, then press Check.", "empty", empty=True)

    # 3. No key configured -> typed fallback.
    if not vision.vision_enabled():
        return _result(
            "", False,
            "Claude vision is off (no ANTHROPIC_API_KEY on the server). Type what you wrote, then Check.",
            "needs_read", needs_read=True,
        )

    shape = answer_shape(answer)
    # 4. Blind transcription.
    try:
        t = vision.transcribe(prepared.png, shape, client)
    except vision.VisionError as e:
        return _result(
            "", False, f"{e.message} You can type what you wrote instead.", "vision_error",
            needs_read=True, error=e.message, error_code=e.code,
        )

    if not t.legible:
        return _result(
            "", False,
            "I couldn't make that out — try writing a bit larger and clearer, or type it instead.",
            "unreadable", needs_read=True, unreadable=True, confidence=t.confidence,
        )

    # 5. Compare in code.
    ok = equivalent(t.text, answer, shape)
    method = "vision"
    # 6. Longer free-text answers: allow a paraphrase if the strict check fails.
    if not ok and shape == "phrase":
        try:
            ok = vision.judge_same_meaning(t.text, answer, client)
            method = "vision+semantic" if ok else "vision"
        except vision.VisionError:
            pass  # judge unavailable: keep the strict result

    extra: dict[str, Any] = {"confidence": t.confidence, "alternatives": t.alternatives}
    return _result(t.text, ok, _default_hint(hint), method, **extra)
