"""OCRmyPDF/Tesseract parser implementation."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from .base import FileParser, UnsupportedFileTypeError

logger = logging.getLogger(__name__)


class OCRMYPDFParser(FileParser):
    """
    Local PDF OCR parser backed by OCRmyPDF and Tesseract.

    OCRmyPDF creates a searchable PDF and optional sidecar text. We prefer
    MarkItDown output from the OCR'd PDF so the parser returns markdown like
    the other local parser, and only fall back to sidecar text when MarkItDown
    cannot extract content from the generated PDF.
    """

    def __init__(
        self,
        *,
        language: str = "eng",
        jobs: int = 1,
        ocrmypdf_module: Any | None = None,
        markitdown: Any | None = None,
    ):
        """Initialize OCRmyPDF parser."""
        language = language.strip()
        if not language:
            raise ValueError("language is required for ocrmypdf parser")
        if jobs < 1:
            raise ValueError("jobs must be >= 1 for ocrmypdf parser")

        try:
            self._ocrmypdf = ocrmypdf_module or __import__("ocrmypdf")
            if markitdown is None:
                from markitdown import MarkItDown

                markitdown = MarkItDown()
        except ImportError as e:
            raise ImportError(
                "ocrmypdf and markitdown packages are required for OCRmyPDF parsing. "
                "Install with: pip install ocrmypdf markitdown"
            ) from e

        self._markitdown = markitdown
        self._language = language
        self._jobs = jobs

    async def convert(self, file_data: bytes, filename: str) -> str:
        """Parse a PDF to markdown using OCRmyPDF followed by MarkItDown."""
        if not self.supports(filename):
            raise UnsupportedFileTypeError(f"ocrmypdf only supports PDF files: '{filename}'")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._convert_sync, bytes(file_data), filename)

    def _convert_sync(self, file_data: bytes, filename: str) -> str:
        """Synchronous OCR and markdown extraction (runs in thread pool)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pdf = tmp_path / f"input{Path(filename).suffix or '.pdf'}"
            output_pdf = tmp_path / "output.pdf"
            sidecar_txt = tmp_path / "sidecar.txt"
            input_pdf.write_bytes(file_data)

            try:
                self._ocrmypdf.ocr(
                    str(input_pdf),
                    str(output_pdf),
                    sidecar=str(sidecar_txt),
                    language=self._language,
                    jobs=self._jobs,
                    skip_text=True,
                    progress_bar=False,
                )

                content = self._extract_markdown_or_sidecar(output_pdf, sidecar_txt)
                if content:
                    return content
                raise RuntimeError(f"No content extracted from '{filename}'")

            except Exception as e:
                logger.error(f"OCRmyPDF parsing failed for {filename}: {e}")
                raise RuntimeError(f"Failed to parse '{filename}' with OCRmyPDF: {e}") from e

    def _extract_markdown_or_sidecar(self, output_pdf: Path, sidecar_txt: Path) -> str:
        """Extract markdown from the OCR'd PDF, falling back to sidecar text."""
        try:
            result = self._markitdown.convert(str(output_pdf))
            markdown = getattr(result, "text_content", None)
            if markdown and markdown.strip():
                return markdown
        except Exception as e:
            logger.warning(f"MarkItDown could not extract OCRmyPDF output PDF, trying sidecar text: {e}")

        if sidecar_txt.exists():
            sidecar = sidecar_txt.read_text(encoding="utf-8", errors="replace")
            if sidecar and sidecar.strip():
                return sidecar

        return ""

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """Check if OCRmyPDF supports this file type."""
        return Path(filename).suffix.lower() == ".pdf"

    def name(self) -> str:
        """Get parser name."""
        return "ocrmypdf"
