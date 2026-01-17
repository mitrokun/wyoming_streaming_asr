# file: asr_stream/handler.py

import argparse
import asyncio
import logging
import os
import time
import wave
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
    """Simple Automatic Gain Control for streaming audio."""
    def __init__(self, target_level=0.7, max_gain=30.0, min_gain=1.0):
        self.target_level = target_level
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.current_peak_envelope = 0.05

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0:
            return audio_chunk

        chunk_max = np.max(np.abs(audio_chunk))
        
        if chunk_max > self.current_peak_envelope:
            alpha = 0.5
        else:
            alpha = 0.01

        self.current_peak_envelope = (1 - alpha) * self.current_peak_envelope + alpha * chunk_max
        safe_envelope = max(self.current_peak_envelope, 1e-6)

        target_gain = self.target_level / safe_envelope
        final_gain = np.clip(target_gain, self.min_gain, self.max_gain)

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
        
        # Debug buffer
        self.audio_buffer = bytearray()
        self.save_dir = os.path.dirname(__file__)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
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
        _LOGGER.debug("Audio stream started.")
        self.stream = self.recognizer.create_stream()
        self.last_stable_text = ""
        self.command_recognized = False
        self.check_performed = False
        self.agc = StreamAGC(target_level=0.7, max_gain=30.0)
        self.audio_buffer = bytearray() # Reset buffer
        await self.write_event(TranscriptStart(language=self.language).event())

    async def _finalize_recognition(self, text: str) -> None:
        if self.stream is None:
            return
        final_text = text.strip()
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text).event())
        await self.write_event(TranscriptStop().event())
        self.stream = None
        self.command_recognized = True

    def _check_for_command(self, text: str) -> Optional[str]:
        if not self.sorted_commands:
            return None
        for command in self.sorted_commands:
            if command in text:
                return command
        return None

    def _save_debug_audio(self):
        """Saves the buffered audio to a WAV file."""
        if not self.audio_buffer:
            return
        try:
            filename = f"debug_audio_{int(time.time() * 1000)}.wav"
            filepath = os.path.join(self.save_dir, filename)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(EXPECTED_SAMPLE_RATE)
                wf.writeframes(self.audio_buffer)
            _LOGGER.debug("Saved debug audio to %s", filepath)
        except Exception as e:
            _LOGGER.error("Failed to save debug audio: %s", e)

    async def _handle_audio_chunk(self, audio_chunk_bytes: bytes) -> None:
        if self.stream is None or self.command_recognized:
            return

        try:
            samples_int16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            
            # Apply AGC
            samples_float32 = self.agc.process(samples_float32)

            # Store for debug saving (float32 -> int16)
            debug_int16 = (samples_float32 * 32767).astype(np.int16)
            self.audio_buffer.extend(debug_int16.tobytes())

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
        if self.stream is None or self.command_recognized:
            self._save_debug_audio() # Save even if aborted early
            return

        _LOGGER.debug("End of audio stream.")
        # Padding 0.4s
        tail_padding = np.zeros(int(EXPECTED_SAMPLE_RATE * 0.4), dtype=np.float32)
        
        # Add padding to debug buffer
        padding_int16 = (tail_padding * 32767).astype(np.int16)
        self.audio_buffer.extend(padding_int16.tobytes())

        self.stream.accept_waveform(EXPECTED_SAMPLE_RATE, tail_padding)
        self.stream.input_finished()
        
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        final_text = self.recognizer.get_result(self.stream).strip()
        
        # Save WAV file before closing
        self._save_debug_audio()

        if not self.check_performed:
            matched_command = self._check_for_command(final_text.lower())
            if matched_command:
                _LOGGER.debug("Command matched (final): '%s'", matched_command)
                await self._finalize_recognition(matched_command)
                return

        await self._finalize_recognition(final_text)