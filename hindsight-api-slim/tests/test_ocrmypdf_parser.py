"""Tests for the OCRmyPDF file parser."""

import importlib.util
import shutil
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight_api.engine.parsers.base import UnsupportedFileTypeError
from hindsight_api.engine.parsers.ocrmypdf import OCRMYPDFParser


@dataclass
class FakeOCRCall:
    """Captured OCRmyPDF invocation."""

    input_file: Path
    output_file: Path
    sidecar: Path
    language: str
    jobs: int
    skip_text: bool
    progress_bar: bool


@dataclass
class FakeOCRMYPDF:
    """Small OCRmyPDF-shaped fake used to avoid shelling out in unit tests."""

    sidecar_text: str = "Sidecar OCR text"
    failure: Exception | None = None
    calls: list[FakeOCRCall] = field(default_factory=list)

    def ocr(
        self,
        input_file: str,
        output_file: str,
        *,
        sidecar: str,
        language: str,
        jobs: int,
        skip_text: bool,
        progress_bar: bool,
    ) -> None:
        input_path = Path(input_file)
        output_path = Path(output_file)
        sidecar_path = Path(sidecar)

        if self.failure is not None:
            raise self.failure

        assert input_path.exists()
        output_path.write_bytes(b"%PDF-1.4\n% OCR output\n")
        sidecar_path.write_text(self.sidecar_text, encoding="utf-8")
        self.calls.append(
            FakeOCRCall(
                input_file=input_path,
                output_file=output_path,
                sidecar=sidecar_path,
                language=language,
                jobs=jobs,
                skip_text=skip_text,
                progress_bar=progress_bar,
            )
        )


@dataclass
class FakeMarkItDown:
    """Small MarkItDown-shaped fake."""

    text_content: str | None = "# OCR markdown"
    failure: Exception | None = None
    converted_paths: list[Path] = field(default_factory=list)

    def convert(self, path: str) -> SimpleNamespace:
        pdf_path = Path(path)
        self.converted_paths.append(pdf_path)
        assert pdf_path.exists()
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(text_content=self.text_content)


def test_ocrmypdf_parser_metadata() -> None:
    """OCRMYPDFParser should expose parser metadata without real OCR."""
    parser = OCRMYPDFParser(ocrmypdf_module=FakeOCRMYPDF(), markitdown=FakeMarkItDown())

    assert parser.name() == "ocrmypdf"
    assert parser.supports("scan.pdf")
    assert parser.supports("SCAN.PDF")
    assert not parser.supports("notes.txt")


@pytest.mark.asyncio
async def test_ocrmypdf_rejects_non_pdf() -> None:
    """Explicit use on a non-PDF should report unsupported file type."""
    parser = OCRMYPDFParser(ocrmypdf_module=FakeOCRMYPDF(), markitdown=FakeMarkItDown())

    with pytest.raises(UnsupportedFileTypeError, match="only supports PDF"):
        await parser.convert(b"not a pdf", "notes.txt")


@pytest.mark.asyncio
async def test_ocrmypdf_runs_with_expected_options_and_cleans_temp_files() -> None:
    """Parser should invoke OCRmyPDF with the configured local OCR options."""
    fake_ocr = FakeOCRMYPDF()
    fake_markitdown = FakeMarkItDown(text_content="# Searchable PDF text")
    parser = OCRMYPDFParser(language="eng+fra", jobs=2, ocrmypdf_module=fake_ocr, markitdown=fake_markitdown)

    result = await parser.convert(b"%PDF-1.4\n", "scan.pdf")

    assert result == "# Searchable PDF text"
    assert len(fake_ocr.calls) == 1
    call = fake_ocr.calls[0]
    assert call.language == "eng+fra"
    assert call.jobs == 2
    assert call.skip_text is True
    assert call.progress_bar is False
    assert fake_markitdown.converted_paths == [call.output_file]
    assert not call.input_file.exists()
    assert not call.output_file.exists()
    assert not call.sidecar.exists()


@pytest.mark.asyncio
async def test_ocrmypdf_falls_back_to_sidecar_when_markitdown_is_empty() -> None:
    """Sidecar text should be returned when MarkItDown extracts no content."""
    parser = OCRMYPDFParser(
        ocrmypdf_module=FakeOCRMYPDF(sidecar_text="Sidecar text"),
        markitdown=FakeMarkItDown(text_content=""),
    )

    result = await parser.convert(b"%PDF-1.4\n", "scan.pdf")

    assert result == "Sidecar text"


@pytest.mark.asyncio
async def test_ocrmypdf_falls_back_to_sidecar_when_markitdown_fails() -> None:
    """Sidecar text should still be used if markdown extraction raises."""
    parser = OCRMYPDFParser(
        ocrmypdf_module=FakeOCRMYPDF(sidecar_text="Recovered sidecar text"),
        markitdown=FakeMarkItDown(failure=RuntimeError("bad pdf")),
    )

    result = await parser.convert(b"%PDF-1.4\n", "scan.pdf")

    assert result == "Recovered sidecar text"


@pytest.mark.asyncio
async def test_ocrmypdf_maps_runtime_errors() -> None:
    """OCR/system dependency failures should become parser RuntimeErrors."""
    parser = OCRMYPDFParser(
        ocrmypdf_module=FakeOCRMYPDF(failure=RuntimeError("missing tesseract")),
        markitdown=FakeMarkItDown(),
    )

    with pytest.raises(RuntimeError, match="Failed to parse 'scan.pdf' with OCRmyPDF"):
        await parser.convert(b"%PDF-1.4\n", "scan.pdf")


def _ocrmypdf_runtime_available() -> bool:
    """Return True when both Python and system OCR dependencies are available."""
    return (
        importlib.util.find_spec("ocrmypdf") is not None
        and shutil.which("tesseract") is not None
        and shutil.which("gs") is not None
    )


@pytest.mark.skipif(not _ocrmypdf_runtime_available(), reason="OCRmyPDF/Tesseract/Ghostscript not installed")
@pytest.mark.asyncio
async def test_ocrmypdf_parser_integration_extracts_image_pdf_text() -> None:
    """Optional smoke test for the real OCR stack when local binaries exist."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 400), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    draw.text((80, 140), "Hindsight OCR test", fill="black", font=font)

    buffer = BytesIO()
    image.save(buffer, format="PDF")

    result = await OCRMYPDFParser().convert(buffer.getvalue(), "scan.pdf")

    assert result.strip()
