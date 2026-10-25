"""Fill in the Ink (Study Companion): lecture notes in your handwriting, with blanks to fill and check."""

from .page import render_lesson
from .solver import answers_for_lesson

__all__ = ["render_lesson", "answers_for_lesson"]
