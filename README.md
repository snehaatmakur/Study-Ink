# Study Ink

Your handwriting, turned into study material. One app, three tools, one look:

| Tab | What it does |
|-----|--------------|
| **Diagram → ink** | Upload a lecture slide; Claude extracts the notes and diagram, and it's redrawn with your handwriting glyphs and sketchy strokes. Edit the diagram JSON, download the PNG, or print a combined study sheet / PDF. |
| **Text → ink** | Type or import notes (TXT, PDF, images via OCR) and render them in your handwriting. Paper, ink colour, size, neatness, re-roll, and Smart Quick Mode (auto fill-in-the-blank worksheet). |
| **Fill in the Ink (Study Companion)** | Lecture notes in *your* handwriting with blanks to fill. Write the answer with a mouse or Apple Pencil, press **Check**, Claude reads your ink, and the answer is checked **in code**. Redraw mode, computed graph paths, PDF + JSON export. |

**Use my handwriting** (top right) walks you through printing the template, filling it in, and uploading it to create your own profile.

## Quick start

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env      # then paste your ANTHROPIC_API_KEY into .env
python3 app.py            # http://localhost:8000
```

`python3 -m studyalong` is the same as `python3 app.py`. Set `PORT` in `.env` to change the port.

Without an API key everything still runs; slide analysis is unavailable and Study Companion falls back to
typed answers (the badge in the top bar says **Vision off · typed answers**).

## Fill in the Ink: how Check works

(`studyalong/vision.py`, `studyalong/grade.py`)

- The browser canvas is transparent with thin green strokes, which models see as near-invisible. The server flattens it to dark ink on white, crops to the writing and scales it up before sending.
- Claude **transcribes blind**: it is never told the expected answer.
- Correctness is decided **in code**: strict match for numbers and labels (`R2`, `10`). Longer free-text phrases that fail the strict match get a text-only paraphrase check. Graph answers (costs, next hops) come from **networkx**, never from the LLM.
- No key, an API error, an empty box, or unreadable ink is **not** marked wrong: the UI asks you to type what you wrote instead and shows why.
- Model: `claude-sonnet-5-5` by default for both slide analysis and handwriting reading; override with `STUDYINK_VISION_MODEL` in `.env`.
- Verify end to end: `python3 -m studyalong.vision`, or open `/study/vision/selftest` (about 3 small API calls).

## Demo path (Dijkstra / CS 168)

Lesson: `studyalong/lessons/dijkstra-r3.json`

- Fill "one source to all others"
- Edge weight R1–R4 = `10`
- Forwarding table for A/B (next hop `R2`, costs `5` / `5`), computed by networkx
- Next hop R3→A = `R2`
- Redraw the graph, then Show reference overlay

## HTTP routes

| Route | Purpose |
|-------|---------|
| `GET /` | The app |
| `POST /analyze-slide`, `/render-diagram`, `/render-redrawn`, `/render-over-source` | Diagram → ink |
| `POST /extract`, `/render` | Text → ink |
| `GET /handwriting/template.png\|pdf`, `GET /handwriting/profiles`, `POST /handwriting/upload` | Use my handwriting (onboarding) |
| `GET /study/lessons[/id]`, `POST /study/render`, `/study/grade`, `/study/solve`, `/study/export/pdf`, `GET /study/redraw/<lesson>/<block>` | Fill in the Ink (Study Companion) |
| `GET /study/health`, `GET /study/vision/selftest` | Vision status / self-test |

`POST /study/render` returns blank bounding boxes + block IDs, and the JSON document model (`studyalong.v1`) is the handoff contract for a later Notability integration.

## Layout

| Path | Role |
|------|------|
| `app.py`, `app.html` | Server and single-page UI (all three tabs) |
| `handwriting/` | Glyph extraction, fallback glyphs, `Writer` (text), `DiagramWriter` (diagrams), onboarding |
| `profiles/` | Handwriting profiles (`trisha_full`, `trisha`, `sneha-2`, plus any you create) |
| `studyalong/` | Study Companion: lesson layout, diagrams, solver, vision + grader, export |
| `extractor.py` | Text from TXT / PDF / image files |
| `cli.py`, `run_sample.py`, `test_roundtrip.py`, `examples/` | Engine CLI and samples |
| `tests/` | Grading + server tests (`python3 -m pytest tests`; the Anthropic client is mocked) |

The Python package is still named `studyalong` internally (and the export format `studyalong.v1`) so saved documents keep working.
