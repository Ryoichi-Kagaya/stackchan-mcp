"""Opus audio frame handling — skeleton for Phase 3.

This module will handle:
- Incoming Opus frames from the device (STT pipeline)
- Outgoing Opus frames to the device (TTS pipeline)

For now, binary frames are logged and discarded.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def handle_audio_frame(data: bytes, session_id: str) -> None:
    """Process an incoming binary Opus frame (stub).

    Phase 3 will pipe this into an STT engine.
    """
    logger.debug(
        "audio_frame session=%s bytes=%d (discarded — Phase 3)",
        session_id,
        len(data),
    )


async def send_audio_frame(data: bytes) -> bytes:
    """Prepare an outgoing Opus frame (stub).

    Phase 3 will generate this from a TTS engine.
    """
    return data
