"""ElevenLabs TTS engine for the stackchan-mcp gateway.

Calls /v1/text-to-speech with ``output_format=pcm_16000``, which returns
raw 16 kHz mono signed-16-bit LE PCM directly — no WAV header, no
resampling needed.

Configuration (environment variables, in lookup order):

    STACKCHAN_ELEVENLABS_API_KEY or ELEVENLABS_API_KEY
        ElevenLabs secret key.

    STACKCHAN_ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE_ID
        Voice ID to use for synthesis.

    STACKCHAN_ELEVENLABS_MODEL_ID
        Model ID.  Defaults to ``eleven_turbo_v2_5`` which supports
        Japanese and reliably returns PCM output.  Do **not** use
        ``eleven_v3`` here — it sometimes returns MP3 even when PCM is
        requested, and the gateway has no MP3 decoder.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import TTSEngine

logger = logging.getLogger(__name__)

_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

#: Model with low latency and reliable PCM output.  Override via env.
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0

#: Magic-byte prefixes that identify an MP3 stream.
_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


class ElevenLabsEngine(TTSEngine):
    """Synthesise text via the ElevenLabs REST API.

    Returns 16 kHz mono PCM suitable for direct Opus encoding.
    httpx is imported lazily inside :meth:`synthesize` so the module
    loads cleanly even without the ``[tts]`` extra.
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.getenv("STACKCHAN_ELEVENLABS_API_KEY")
            or os.getenv("ELEVENLABS_API_KEY", "")
        )
        self._voice_id = (
            voice_id
            or os.getenv("STACKCHAN_ELEVENLABS_VOICE_ID")
            or os.getenv("ELEVENLABS_VOICE_ID", "")
        )
        self._model_id = (
            model_id or os.getenv("STACKCHAN_ELEVENLABS_MODEL_ID") or DEFAULT_MODEL_ID
        )
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def synthesize(self, text: str, **opts: Any) -> bytes:
        """POST to ElevenLabs and return 16 kHz mono PCM bytes.

        The ``pcm_16000`` output format is requested so the API returns
        raw PCM with no container overhead.  If the response looks like
        MP3 (magic bytes) a :class:`RuntimeError` is raised — switch to
        a model that supports PCM output (see module docstring).
        """
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is not installed. Install with "
                "'pip install stackchan-mcp[tts]' to enable ElevenLabs."
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise ValueError("ElevenLabs synthesize: 'text' must be a non-empty string")
        if not self._api_key:
            raise RuntimeError(
                "ElevenLabs API key not configured. "
                "Set ELEVENLABS_API_KEY (or STACKCHAN_ELEVENLABS_API_KEY)."
            )
        if not self._voice_id:
            raise RuntimeError(
                "ElevenLabs voice ID not configured. "
                "Set ELEVENLABS_VOICE_ID (or STACKCHAN_ELEVENLABS_VOICE_ID)."
            )

        url = f"{_TTS_URL}/{self._voice_id}?output_format=pcm_16000"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        client_kwargs: dict[str, Any] = {"timeout": self._timeout_seconds}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"ElevenLabs API returned {resp.status_code}: {resp.text[:120]}"
                )
            pcm = resp.content

        if pcm[:3] in _MP3_MAGIC:
            raise RuntimeError(
                f"ElevenLabs model '{self._model_id}' returned MP3 instead "
                "of PCM. Use 'eleven_turbo_v2_5' or 'eleven_flash_v2_5' "
                "(set STACKCHAN_ELEVENLABS_MODEL_ID)."
            )

        logger.info(
            "ElevenLabs synthesised %d bytes PCM (16 kHz) voice=%s model=%s text=%r",
            len(pcm),
            self._voice_id,
            self._model_id,
            text[:60],
        )
        return pcm
