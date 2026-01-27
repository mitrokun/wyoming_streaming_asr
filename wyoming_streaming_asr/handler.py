import argparse
import asyncio
import logging
from typing import Set, List, Optional

import numpy as np
from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from sherpa_onnx import OnlineRecognizer

_LOGGER = logging.getLogger(__name__)
EXPECTED_SAMPLE_RATE = 16000


class StreamAGC:
    """
    Simple Automatic Gain Control with Auto-Calibration.
    Detects if the source is already normalized (DSP) or quiet (raw mic).
    """
    def __init__(self, target_level=0.6, max_gain=30.0, min_gain=1.0):
        self.target_level = target_level
        self.absolute_max_gain = max_gain
        self.min_gain = min_gain
        
        # Start with target_level to prevent noise bursts during calibration
        self.current_peak_envelope = target_level 

        # Calibration state
        self.calib_peak = 0.0
        self.calib_frames = 12
        self.is_calibrated = False
        
        # Threshold to detect pre-processed signals (-30dB for RespeakerLite)
        # Record audio and check the silence level on your device 
        self.dsp_threshold = 0.031 
        
        # Default to safe mode (gain x1.0) until proven otherwise
        self.active_max_gain = 1.0 

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0:
            return audio_chunk

        chunk_max = np.max(np.abs(audio_chunk))

        # === 1. Calibration Phase ===
        if self.calib_frames > 0:
            self.calib_peak = max(self.calib_peak, chunk_max)
            self.calib_frames -= 1
            return audio_chunk

        # === 2. Source Type Detection (Run once) ===
        if not self.is_calibrated:
            if self.calib_peak > self.dsp_threshold:
                # Loud source (DSP detected): Disable amplification
                self.active_max_gain = 1.0
            else:
                # Quiet source (Raw Mic): Enable full gain
                self.active_max_gain = self.absolute_max_gain
                # Reset envelope to low value for immediate reaction
                self.current_peak_envelope = 0.1
            
            self.is_calibrated = True

        # === 3. DSP Bypass ===
        if self.active_max_gain <= 1.0:
            return np.clip(audio_chunk, -1.0, 1.0)

        # === 4. AGC Logic ===
        # Fast attack, slow release
        if chunk_max > self.current_peak_envelope:
            alpha = 0.5
        else:
            alpha = 0.01

        self.current_peak_envelope = (1 - alpha) * self.current_peak_envelope + alpha * chunk_max
        safe_envelope = max(self.current_peak_envelope, 1e-6)

        target_gain = self.target_level / safe_envelope
        final_gain = np.clip(target_gain, self.min_gain, self.active_max_gain)

        return np.tanh(audio_chunk * final_gain)


class SherpaOnnxEventHandler(AsyncEventHandler):
    """Event handler for each client using sherpa-onnx."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        recognizer: OnlineRecognizer,
        commands: Set[str],
        command_max_words: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.cli_args = cli_args
        self.recognizer = recognizer
        self.sorted_commands: List[str] = sorted(list(commands), key=len, reverse=True)
        self.command_max_words = command_max_words
        self.language = self.cli_args.language
        self.stream = None
        self.last_stable_text = ""
        self.command_recognized = False
        self.check_performed = False
        self.agc = StreamAGC(target_level=0.7, max_gain=30.0)

    async def handle_event(self, event: Event) -> bool:
        """Main method for handling incoming events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent server information (info)")
            return True
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            if audio_start.rate != EXPECTED_SAMPLE_RATE:
                _LOGGER.warning("Unexpected sample rate: %s", audio_start.rate)
            await self._handle_audio_start()
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            await self._handle_audio_chunk(chunk.audio)
            return True
        if AudioStop.is_type(event.type):
            await self._handle_audio_stop()
            return False
        if Error.is_type(event.type):
            _LOGGER.error("Received error from client: %s", event.text)
        return True

    async def _handle_audio_start(self) -> None:
        """Resets the state for a new phrase."""
        _LOGGER.debug("Audio stream started.")
        self.stream = self.recognizer.create_stream()
        self.last_stable_text = ""
        self.command_recognized = False
        self.check_performed = False
        # Reset AGC for new phrase
        self.agc = StreamAGC(target_level=0.7, max_gain=30.0)
        await self.write_event(TranscriptStart(language=self.language).event())

    async def _finalize_recognition(self, text: str) -> None:
        """Sends the final result and ends the session."""
        if self.stream is None:
            return
        final_text = text.strip()
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text).event())
        await self.write_event(TranscriptStop().event())
        self.stream = None
        self.command_recognized = True

    def _check_for_command(self, text: str) -> Optional[str]:
        """
        Searches for the longest matching command in the text.
        Returns the matched command string or None.
        """
        if not self.sorted_commands:
            return None
            
        for command in self.sorted_commands:
            if command in text:
                return command
        return None

    async def _handle_audio_chunk(self, audio_chunk_bytes: bytes) -> None:
        """Processes an audio chunk and triggers a check upon reaching the stable word threshold."""
        if self.stream is None or self.command_recognized:
            return

        try:
            samples_int16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            
            # Apply AGC
            samples_float32 = self.agc.process(samples_float32)

            self.stream.accept_waveform(EXPECTED_SAMPLE_RATE, samples_float32)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            
            current_full_text = self.recognizer.get_result(self.stream).strip().lower()

            if not current_full_text:
                return
            
            words = current_full_text.split()
            stable_words = words[:-1] if len(words) > 1 else words

            if self.sorted_commands and not self.check_performed and self.command_max_words > 0:
                if len(stable_words) >= self.command_max_words:
                    self.check_performed = True
                    
                    matched_command = self._check_for_command(current_full_text)
                    if matched_command:
                        _LOGGER.debug("Command matched (stream): '%s'", matched_command)
                        await self._finalize_recognition(matched_command)
                        return

            if not self.command_recognized:
                current_stable_text = " ".join(stable_words)
                if current_stable_text and current_stable_text != self.last_stable_text:
                    delta_text = current_stable_text[len(self.last_stable_text) :].strip()
                    if delta_text:
                        _LOGGER.debug("Δ: '%s'", delta_text)
                        await self.write_event(TranscriptChunk(text=delta_text).event())
                    self.last_stable_text = current_stable_text

        except Exception as e:
            _LOGGER.exception("Error processing audio chunk")
            await self.write_event(Error(text=str(e)).event())

    async def _handle_audio_stop(self) -> None:
        """Finalizes recognition and performs a final check if one hasn't been done yet."""
        if self.stream is None or self.command_recognized:
            return

        _LOGGER.debug("End of audio stream.")
        # Padding 0.3s
        tail_padding = np.zeros(int(EXPECTED_SAMPLE_RATE * 0.3), dtype=np.float32)
        self.stream.accept_waveform(EXPECTED_SAMPLE_RATE, tail_padding)
        self.stream.input_finished()
        
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        final_text = self.recognizer.get_result(self.stream).strip()
        
        _LOGGER.debug("Full final text for checking: '%s'", final_text)

        if not self.check_performed:
            matched_command = self._check_for_command(final_text.lower())
            if matched_command:
                _LOGGER.debug("Command matched (final): '%s'", matched_command)
                await self._finalize_recognition(matched_command)
                return

        await self._finalize_recognition(final_text)

