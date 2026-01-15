# Wyoming server for streaming ASR

Server uses the [sherpa_onnx](https://github.com/k2-fsa/sherpa-onnx) project to run models on the zipformer architecture. Available models can be found [here](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).

Command dictionary [optional] - if specified, after recognizing `command-max-words` words, a dictionary check is performed, and if a match is found, it will become the final recognition result. The function works in conjunction with this custom [client](https://github.com/mitrokun/asr_proxy/tree/extended) (it is capable of stopping audio transmission on command from the server).

The model directory must contain 4 files: `encoder.onnx decoder.onnx joiner.onnx tokens.txt `

Required libraries: `pip install sherpa-onnx wyoming numpy`

Example launch for Windows:
```
python -m wyoming_streaming_asr --model-dir "\models\vosk-ru" --language ru --command-file commands.txt --command-max-words 5
```


Latest Russian models from Alpha Cephei https://huggingface.co/alphacep/

The streaming version of Parakeet (en) has been [released](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14.tar.bz2). Run this model with the `--decoding-method greedy_search` option.
