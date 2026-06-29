"""xiaozhi AI agent mode conversation loop.

Drives STT → familiar-ai → TTS for a single device session when the device
connects in xiaozhi AI agent mode (hello includes ``audio_params``).

The device side drives the conversation boundary:
  - ``{"type":"listen","state":"start"}`` → we open a recording slot
  - binary Opus frames flow up the WebSocket (buffered by audio_stream)
  - ``{"type":"listen","state":"stop"}``  → we close the slot and run STT
  then call familiar-ai, synthesise TTS, and push audio back.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .audio_stream import start_recording, stop_recording
from .familiar_client import call_familiar
from .stt.audio_utils import decode_opus_frames
from .stt.base import get_registry as get_stt_registry
from .tts.audio_utils import (
    DEVICE_FRAME_DURATION_MS,
    encode_opus_frames,
)
from .tts.base import get_registry as get_tts_registry
from .tts.orchestrator import TTS_START_TRANSITION_DELAY_S

if TYPE_CHECKING:
    from .esp32_client import ESP32Manager

logger = logging.getLogger(__name__)

DEFAULT_STT_ENGINE = "faster-whisper"
DEFAULT_TTS_ENGINE = "voicevox"
DEFAULT_STT_LANGUAGE = "ja"

# Max time to wait for listen.stop after listen.start.
# If the firmware sends no stop (e.g. firmware bug), we time out and
# treat whatever frames accumulated as the utterance.
LISTEN_STOP_TIMEOUT_S = 30.0


def _appraisal_to_emotion(valence: float, arousal: float) -> str:
    """Map familiar-ai appraisal (valence, arousal) to a xiaozhi emotion name."""
    if valence >= 0 and arousal >= 0.5:
        return "happy"
    if valence >= 0 and arousal < 0.5:
        return "neutral"
    if valence < 0 and arousal >= 0.5:
        return "angry"
    return "sad"


class ConversationManager:
    """Manages the xiaozhi AI agent mode conversation loop for one device session.

    The loop waits for ``listen.start`` / ``listen.stop`` events dispatched
    by :class:`~stackchan_mcp.esp32_client.ESP32Manager._handler`, runs STT,
    calls familiar-ai, and plays back TTS — in a continuous cycle until the
    device disconnects.
    """

    def __init__(self, esp32: "ESP32Manager", familiar_url: str, session_id: str) -> None:
        self._esp32 = esp32
        self._familiar_url = familiar_url
        self._session_id = session_id
        self._listen_start_event: asyncio.Event = asyncio.Event()
        self._listen_stop_event: asyncio.Event = asyncio.Event()
        self._stopped = False

    @property
    def session_id(self) -> str:
        """The ESP32 connection session this conversation is bound to.

        The handler compares this against the inbound ``listen`` message's
        session so a stale conversation cannot consume events for a device
        that has since reconnected with a fresh session id.
        """
        return self._session_id

    # ------------------------------------------------------------------
    # Public event callbacks (called from the WebSocket message loop)
    # ------------------------------------------------------------------

    def on_listen_start(self, session_id: str) -> None:
        """Called when the device sends ``{"type":"listen","state":"start"}``."""
        if session_id != self._session_id:
            return
        start_recording(session_id)
        self._listen_stop_event.clear()
        self._listen_start_event.set()
        logger.debug("conversation: listen.start session=%s", session_id)

    def on_listen_stop(self, session_id: str) -> None:
        """Called when the device sends ``{"type":"listen","state":"stop"}``."""
        if session_id != self._session_id:
            return
        self._listen_stop_event.set()
        logger.debug("conversation: listen.stop session=%s", session_id)

    def stop(self) -> None:
        """Signal the loop to exit on next iteration."""
        self._stopped = True
        self._listen_start_event.set()  # unblock run() if it's waiting

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Conversation loop: runs until :meth:`stop` is called."""
        logger.info(
            "conversation: loop started session=%s familiar=%s",
            self._session_id,
            self._familiar_url,
        )
        while not self._stopped:
            self._listen_start_event.clear()
            await self._listen_start_event.wait()
            if self._stopped:
                break
            try:
                await self._run_turn()
            except Exception as exc:
                logger.error("conversation: turn failed: %s", exc, exc_info=True)
        logger.info("conversation: loop stopped session=%s", self._session_id)

    # ------------------------------------------------------------------
    # Single turn: STT → familiar-ai → TTS
    # ------------------------------------------------------------------

    async def _run_turn(self) -> None:
        # Wait for listen.stop (device signals end of speech)
        try:
            await asyncio.wait_for(
                self._listen_stop_event.wait(),
                timeout=LISTEN_STOP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "conversation: listen.stop not received within %.0fs; "
                "processing accumulated frames",
                LISTEN_STOP_TIMEOUT_S,
            )
        self._listen_stop_event.clear()

        frames = stop_recording()
        if not frames:
            logger.info("conversation: no audio frames captured — skipping turn")
            return

        # --- STT ---
        try:
            text = await self._stt(frames)
        except Exception as exc:
            logger.error("conversation: STT failed: %s", exc)
            return

        if not text.strip():
            logger.info("conversation: empty transcription — skipping turn")
            return
        logger.info("conversation: STT result=%r", text[:120])

        # Show user speech bubble while familiar-ai processes the reply
        try:
            await self._esp32.send_stt_text(text)
        except Exception as exc:
            logger.warning("conversation: failed to send STT display text: %s", exc)

        # --- familiar-ai ---
        try:
            result = await call_familiar(text, self._familiar_url)
        except Exception as exc:
            logger.error("conversation: familiar-ai call failed: %s", exc)
            return

        reply = result.get("text", "")
        emotion = result.get("emotion", "neutral")

        if not reply.strip():
            logger.info("conversation: familiar-ai returned empty reply — skipping TTS")
            return
        logger.info("conversation: familiar-ai reply=%r emotion=%s", reply[:80], emotion)

        # --- TTS ---
        try:
            await self._tts_and_send(reply, emotion)
        except Exception as exc:
            logger.error("conversation: TTS failed: %s", exc)

    # ------------------------------------------------------------------
    # STT helper
    # ------------------------------------------------------------------

    async def _stt(self, frames: list[bytes]) -> str:
        pcm = decode_opus_frames(frames)
        if not pcm:
            return ""
        registry = get_stt_registry()
        engine = registry.get(DEFAULT_STT_ENGINE)
        if engine is None:
            raise RuntimeError(
                f"STT engine '{DEFAULT_STT_ENGINE}' not registered. "
                "Install stackchan-mcp[stt] to enable faster-whisper."
            )
        result = await engine.transcribe(pcm, language=DEFAULT_STT_LANGUAGE)
        return result.get("text", "")

    # ------------------------------------------------------------------
    # TTS helper
    # ------------------------------------------------------------------

    async def _tts_and_send(self, text: str, emotion: str) -> None:
        registry = get_tts_registry()
        engine = registry.get(DEFAULT_TTS_ENGINE)
        if engine is None:
            raise RuntimeError(
                f"TTS engine '{DEFAULT_TTS_ENGINE}' not registered. "
                "Install stackchan-mcp[tts] and ensure VOICEVOX is running."
            )

        pcm = await engine.synthesize(text)
        if not pcm:
            logger.warning("conversation: TTS engine produced no PCM")
            return

        opus_frames = list(encode_opus_frames(pcm))
        if not opus_frames:
            logger.warning("conversation: Opus encoding produced no frames")
            return

        esp32 = self._esp32
        tts_lock = esp32.tts_lock

        async with tts_lock:
            # Update the display face before audio starts.
            # llm.emotion is the only message the firmware actually reads for
            # SetEmotion(); the emotion field in tts.start is ignored.
            await esp32.send_llm_emotion(emotion)
            # tts.start transitions the device into kDeviceStateSpeaking so
            # it begins accepting Opus frames from the decode queue.
            await esp32.send_tts_state("start", emotion=emotion)
            await esp32.send_tts_state("sentence_start", text=text)
            await asyncio.sleep(TTS_START_TRANSITION_DELAY_S)

            # Push Opus frames at the device's consumption rate to avoid
            # overflowing the firmware's decode queue (~40 frames).
            frame_period_s = DEVICE_FRAME_DURATION_MS / 1000.0
            loop = asyncio.get_event_loop()
            next_send_time = loop.time()
            push_error: Exception | None = None
            sent = 0
            try:
                for frame in opus_frames:
                    now = loop.time()
                    if now < next_send_time:
                        await asyncio.sleep(next_send_time - now)
                    try:
                        await esp32.send_audio_frame(frame)
                    except ConnectionError as exc:
                        push_error = exc
                        break
                    sent += 1
                    next_send_time += frame_period_s
            finally:
                try:
                    await esp32.send_tts_state("stop")
                except ConnectionError:
                    pass

        if push_error is not None:
            raise RuntimeError(
                f"Device disconnected after {sent}/{len(opus_frames)} frames: {push_error}"
            ) from push_error

        logger.info(
            "conversation: TTS sent frames=%d duration_ms=%d",
            sent,
            sent * DEVICE_FRAME_DURATION_MS,
        )
