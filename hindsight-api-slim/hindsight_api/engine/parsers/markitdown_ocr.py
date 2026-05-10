"""MarkItDown OCR parser implementation."""

import asyncio
import base64
import logging
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import FileParser

logger = logging.getLogger(__name__)


class _GeminiChatCompletionsAdapter:
    """Small OpenAI-shaped adapter backed by the native Google Gen AI SDK."""

    def __init__(self, client: Any):
        self._client = client

    def create(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> SimpleNamespace:
        from google.genai import types as genai_types

        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                if content:
                    parts.append(genai_types.Part.from_text(text=content))
                continue

            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        parts.append(genai_types.Part.from_text(text=text))
                elif item.get("type") == "image_url":
                    data_uri = (item.get("image_url") or {}).get("url")
                    image_part = _part_from_data_uri(data_uri)
                    if image_part is not None:
                        parts.append(image_part)

        response = self._client.models.generate_content(model=model, contents=parts)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response.text or ""),
                )
            ]
        )


class _GeminiChatAdapter:
    """OpenAI-shaped ``client.chat`` namespace for markitdown-ocr."""

    def __init__(self, client: Any):
        self.completions = _GeminiChatCompletionsAdapter(client)


class _GeminiOpenAIShapeClient:
    """Enough of the OpenAI client shape for markitdown-ocr's vision service."""

    def __init__(self, *, api_key: str):
        from google import genai

        self.chat = _GeminiChatAdapter(genai.Client(api_key=api_key))


def _part_from_data_uri(data_uri: str | None) -> Any | None:
    """Convert an OpenAI image_url data URI to a Gemini Part."""
    if not data_uri:
        return None

    match = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", data_uri, flags=re.DOTALL)
    if not match:
        return None

    from google.genai import types as genai_types

    return genai_types.Part.from_bytes(
        data=base64.b64decode(match.group("data")),
        mime_type=match.group("mime"),
    )


def _create_llm_client(
    *,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
    default_headers: dict[str, str] | None,
) -> Any:
    """Create the LLM client shape expected by markitdown-ocr."""
    if (provider or "").lower() == "gemini":
        if not api_key:
            raise ValueError("api_key is required for markitdown_ocr parser when provider='gemini'")
        return _GeminiOpenAIShapeClient(api_key=api_key)

    from openai import OpenAI

    client_kwargs: dict[str, object] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif base_url:
        # Local OpenAI-compatible servers such as LM Studio/Ollama often
        # ignore the API key but the OpenAI SDK still requires one.
        client_kwargs["api_key"] = "not-needed"
    if base_url:
        client_kwargs["base_url"] = base_url
    if default_headers:
        client_kwargs["default_headers"] = default_headers

    return OpenAI(**client_kwargs)


class MarkitdownOCRParser(FileParser):
    """
    MarkItDown parser with the markitdown-ocr plugin enabled.

    This parser uses MarkItDown's plugin system plus a configured vision model
    to OCR scanned pages/images embedded in supported documents.
    It is intentionally exposed as a separate parser from ``markitdown`` so
    callers opt into LLM cost and latency explicitly.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize MarkItDown with OCR plugin support."""
        if not model:
            raise ValueError("model is required for markitdown_ocr parser")

        try:
            import markitdown_ocr  # noqa: F401 - imported to ensure plugin package is installed
            from markitdown import MarkItDown
        except ImportError as e:
            raise ImportError(
                "markitdown-ocr package is required for OCR file parsing. "
                "Install with: pip install 'markitdown-ocr[llm]'"
            ) from e

        self._markitdown = MarkItDown(
            enable_plugins=True,
            llm_client=_create_llm_client(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            ),
            llm_model=model,
            llm_prompt=prompt,
        )

    async def convert(self, file_data: bytes, filename: str) -> str:
        """Parse file to markdown using MarkItDown with OCR plugin support."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._convert_sync, file_data, filename)

    def _convert_sync(self, file_data: bytes, filename: str) -> str:
        """Synchronous parsing (runs in thread pool)."""
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            result = self._markitdown.convert(tmp_path)

            if not result or not result.text_content:
                raise RuntimeError(f"No content extracted from '{filename}'")

            return result.text_content

        except Exception as e:
            logger.error(f"MarkItDown OCR parsing failed for {filename}: {e}")
            raise RuntimeError(f"Failed to parse '{filename}' with OCR: {e}") from e

        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """Check if MarkItDown OCR supports this file type."""
        supported_extensions = {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
        }

        ext = Path(filename).suffix.lower()
        return ext in supported_extensions

    def name(self) -> str:
        """Get parser name."""
        return "markitdown_ocr"
