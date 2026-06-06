"""HTTP client for familiar-ai voice_turn endpoint.

Sends transcribed text to familiar-ai and returns the reply text and emotion.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0


async def call_familiar(text: str, url: str) -> dict[str, Any]:
    """POST ``text`` to familiar-ai and return ``{"text": ..., "emotion": ...}``.

    Args:
        text: Transcribed user speech.
        url:  Full URL of the ``/voice_turn`` endpoint
              (e.g. ``http://localhost:8090/voice_turn``).

    Returns:
        Dict with at least ``text`` (str) and ``emotion`` (str) keys.
        ``emotion`` is one of ``happy``, ``neutral``, ``sad``, ``angry``.

    Raises:
        RuntimeError: on HTTP error or network failure.
    """
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json={"text": text}) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"familiar-ai returned HTTP {resp.status}: {body[:200]}"
                    )
                data = await resp.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"familiar-ai request failed: {exc}") from exc

    logger.debug("familiar_client: reply=%r emotion=%s", str(data.get("text", ""))[:80], data.get("emotion"))
    return data
