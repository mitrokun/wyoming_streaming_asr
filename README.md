# Wyoming server for streaming ASR

Server uses the [sherpa_onnx](https://github.com/k2-fsa/sherpa-onnx) project to run models on the zipformer architecture. Available models can be found [here](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).

Command dictionary [optional] - if specified, after recognizing `command-max-words` words, a dictionary check is performed, and if a match is found, it will become the final recognition result.

The model directory must contain 4 files: `encoder.onnx decoder.onnx joiner.onnx tokens.txt `

Example launch for Windows:
```
python -m wyoming_streaming_asr --model-dir "\models\vosk-ru" --language ru --debug --command-file commands.txt --command-max-words 5
```
