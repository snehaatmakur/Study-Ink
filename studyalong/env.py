"""Tiny .env loader (no extra dependency).

The API process reads ANTHROPIC_API_KEY from the real environment first, then from a `.env`
file in the project root (or the current directory). Existing environment variables always win.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv() -> list[str]:
    """Load KEY=VALUE lines from .env files. Returns the paths that were read."""
    loaded: list[str] = []
    seen: set[str] = set()
    for base in (ROOT, os.getcwd()):
        path = os.path.join(base, ".env")
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if key and val and key not in os.environ:
                    os.environ[key] = val
        loaded.append(path)
    return loaded
