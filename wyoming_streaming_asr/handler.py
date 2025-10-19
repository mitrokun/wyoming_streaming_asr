# file: asr_stream/handler.py

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
        self.check_performed = False # Flag that the one-time check has been performed
        _LOGGER.debug("Event handler initialized")

    async def handle_event(self, event: Event) -> bool:
        """Main method for handling incoming events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent server information (info)")
            return True
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            if audio_start.rate != EXPECTED_SAMPLE_RATE:
                _LOGGER.warning(
                    "Unexpected sample rate: %s. The model expects %s.",
                    audio_start.rate,
                    EXPECTED_SAMPLE_RATE,
                )
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
        _LOGGER.debug("Audio stream started. Creating new recognition stream.")
        self.stream = self.recognizer.create_stream()
        self.last_stable_text = ""
        self.command_recognized = False
        self.check_performed = False # Reset the flag
        await self.write_event(TranscriptStart(language=self.language).event())
        _LOGGER.debug("Sent TranscriptStart.")

    async def _finalize_recognition(self, text: str) -> None:
        """Sends the final result and ends the session."""
        if self.stream is None:
            return
        final_text = text.strip()
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text).event())
        await self.write_event(TranscriptStop().event())
        _LOGGER.debug("Sent final Transcript and TranscriptStop.")
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
            self.stream.accept_waveform(EXPECTED_SAMPLE_RATE, samples_float32)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            
            current_full_text = self.recognizer.get_result(self.stream).strip().lower()

            if not current_full_text:
                return
            
            words = current_full_text.split()
            stable_words = words[:-1] if len(words) > 1 else words

            # --- ONE-TIME CHECK LOGIC ---
            if self.sorted_commands and not self.check_performed and self.command_max_words > 0:
                stable_word_count = len(stable_words)
                if stable_word_count >= self.command_max_words:
                    _LOGGER.debug("Stable word count threshold of %s reached, triggering one-time check.", self.command_max_words)
                    self.check_performed = True
                    
                    matched_command = self._check_for_command(current_full_text)
                    if matched_command:
                        _LOGGER.debug("Command '%s' matched during streaming. Stopping immediately.", matched_command)
                        await self._finalize_recognition(matched_command)
                        return

            # --- INTERMEDIATE RESULTS LOGIC ---
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

        _LOGGER.debug("End of audio stream. Getting final result.")
        tail_padding = np.zeros(int(EXPECTED_SAMPLE_RATE * 0.4), dtype=np.float32)
        self.stream.accept_waveform(EXPECTED_SAMPLE_RATE, tail_padding)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        final_text = self.recognizer.get_result(self.stream).strip()
        
        # --- NEW DEBUG LOG ---
        _LOGGER.debug("Full final text for checking: '%s'", final_text)

        if not self.check_performed:
            _LOGGER.debug("Performing final command check at the end of speech.")
            matched_command = self._check_for_command(final_text.lower())
            if matched_command:
                # --- IMPROVED LOG MESSAGE ---
                _LOGGER.debug("Command '%s' matched at the end of speech.", matched_command)
                await self._finalize_recognition(matched_command)
                return

        # If no command was ever found, send the full text.
        await self._finalize_recognition(final_text)