import base64
import io
import os
import tempfile
from pathlib import Path
from PIL import Image


def extract_text_from_file(file_path):
    """Extracts text from .txt, .pdf, .png files safely."""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            import fitz  # PyMuPDF

            extracted_pages = []
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()

                    # Fallback to OCR if page has little/no text
                    if len(text) < 30:
                        try:
                            import pytesseract

                            pix = page.get_pixmap(dpi=200)
                            img = Image.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples
                            )
                            ocr_text = pytesseract.image_to_string(img).strip()
                            if ocr_text:
                                text = ocr_text
                        except Exception:
                            pass  # Keep whatever standard text was extracted

                    if text:
                        extracted_pages.append(
                            f"--- Slide / Page {page_num} ---\n{text}"
                        )

            return "\n\n".join(extracted_pages)

        elif ext == ".png":
            try:
                import pytesseract

                image = Image.open(file_path)
                text = pytesseract.image_to_string(image).strip()
                return text if text else "No text found in image."
            except Exception as e:
                return (
                    f"Error running OCR: {e}. Ensure 'tesseract' is installed."
                )

        else:
            return f"Error: Unsupported file format {ext}"

    except Exception as e:
        return f"Error processing {file_path}: {e}"


def extract_text_from_bytes(raw_bytes, filename):
    """Extracts text from raw byte data given a filename or extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".png"}:
        raise ValueError("Choose a TXT, PDF, or PNG file")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(raw_bytes)
        temp.flush()
        temp_path = temp.name

    try:
        extracted = extract_text_from_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if extracted.startswith("Error:"):
        raise ValueError(extracted)

    return extracted