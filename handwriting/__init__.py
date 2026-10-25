from .font import Glyph, HandwritingFont
from .render import Writer
from .diagram import DiagramWriter

__all__ = ["Glyph", "HandwritingFont", "extract_font", "ExtractionError", "add_fallbacks", "Writer", "DiagramWriter"]


def __getattr__(name):
    # Rendering an already-saved profile does not need the CV extraction stack.
    # Load those optional dependencies only when their APIs are requested.
    if name in {"extract_font", "ExtractionError"}:
        from . import extract
        return getattr(extract, name)
    if name == "add_fallbacks":
        from .fallback import add_fallbacks
        return add_fallbacks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
