import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Set

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from sherpa_onnx import OnlineRecognizer

from .handler import SherpaOnnxEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the sherpa-onnx model directory (tokens.txt, encoder.onnx, etc.)",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Default language for transcription",
    )
    parser.add_argument(
        "--uri", default="tcp://0.0.0.0:10303", help="URI for the server to listen on"
    )
    parser.add_argument(
        "--hotwords-file",
        default="",
        help="Path to a hotwords.txt file with a list of commands to improve accuracy.",
    )
    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=3.0,
        help="Boosting score for hotwords.",
    )
    parser.add_argument(
        "--decoding-method",
        default="modified_beam_search",
        choices=["greedy_search", "modified_beam_search"],
        help="Decoding method. Use 'greedy_search' for Parakeet/NeMo models, 'modified_beam_search' for Zipformer.",
    )
    parser.add_argument(
        "--bpe-vocab",
        default="",
        help="Path to the model's bpe.vocab or unigram.vocab file (required for hotwords).",
    )
    # Early stopping arguments
    parser.add_argument(
        "--command-file",
        default="",
        help="Path to a file with commands for early stopping. If not provided, the feature is disabled.",
    )
    parser.add_argument(
        "--command-max-words",
        type=int,
        default=5,
        help="Word count threshold to trigger a one-time command check (for command-file).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _LOGGER.debug(args)

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        _LOGGER.critical("Model directory not found: %s", model_dir)
        sys.exit(1)

    # Load commands from file
    commands: Set[str] = set()
    if args.command_file:
        command_path = Path(args.command_file)
        try:
            with open(command_path, "r", encoding="utf-8") as f:
                commands = {line.strip().lower() for line in f if line.strip()}
            _LOGGER.info("Loaded %s command(s) for early stopping from %s", len(commands), command_path)
        except Exception as e:
            _LOGGER.warning("Failed to read command file %s: %s", command_path, e)
    
    # Load model
    try:
        _LOGGER.info("Loading sherpa-onnx model from %s", model_dir)
        
        tokens_path = model_dir / "tokens.txt"
        encoder_path = model_dir / "encoder.int8.onnx"
        if not encoder_path.exists():
             encoder_path = model_dir / "encoder.onnx"
        decoder_path = model_dir / "decoder.int8.onnx"
        if not decoder_path.exists():
             decoder_path = model_dir / "decoder.onnx"
        joiner_path = model_dir / "joiner.int8.onnx"
        if not joiner_path.exists():
             joiner_path = model_dir / "joiner.onnx"

        for p in [tokens_path, encoder_path, decoder_path, joiner_path]:
             if not p.exists():
                  _LOGGER.critical("Required model file not found: %s", p)
                  sys.exit(1)

        if args.hotwords_file and not args.bpe_vocab:
            _LOGGER.critical("--bpe-vocab is required when using --hotwords-file.")
            sys.exit(1)

        recognizer = OnlineRecognizer.from_transducer(
            tokens=str(tokens_path),
            encoder=str(encoder_path),
            decoder=str(decoder_path),
            joiner=str(joiner_path),
            num_threads=4,
            sample_rate=16000,
            decoding_method=args.decoding_method,
            hotwords_file=args.hotwords_file,
            hotwords_score=args.hotwords_score,
            bpe_vocab=args.bpe_vocab,
            modeling_unit="bpe",
        )
        _LOGGER.info("Model loaded successfully.")
    except Exception as e:
        _LOGGER.exception("Error loading model")
        sys.exit(1)

    # Wyoming server info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="sherpa-onnx",
                description="Fast, local speech recognition with sherpa-onnx",
                attribution=Attribution(
                    name="k2-fsa", url="https://github.com/k2-fsa/sherpa-onnx"
                ),
                installed=True,
                supports_transcript_streaming=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=model_dir.name,
                        description=f"sherpa-onnx model {model_dir.name}",
                        attribution=Attribution(
                            name="k2-fsa",
                            url="https://github.com/k2-fsa/sherpa-onnx",
                        ),
                        installed=True,
                        languages=[args.language],
                        version="1.0",
                    )
                ],
            )
        ],
    )

    # Start server
    _LOGGER.info("Server is ready and listening at %s", args.uri)

    handler_factory = partial(
        SherpaOnnxEventHandler,
        wyoming_info=wyoming_info,
        cli_args=args,
        recognizer=recognizer,
        commands=commands,
        command_max_words=args.command_max_words,
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
