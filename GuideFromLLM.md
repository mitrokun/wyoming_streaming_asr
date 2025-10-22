# Quick Guide to Running wyoming_streaming_asr for Beginners (Windows and Linux)

The model directory (`models/kroko`) is in the repository and contains `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`.

## Prerequisites
- Python 3.11–3.13 installed ([python.org](https://www.python.org/downloads/)).
- Git installed ([git-scm.com](https://git-scm.com/downloads/)).

## Step 1: Download the Project
1. Open a terminal:
   - **Windows**: CMD (Win + R, type `cmd`).
   - **Linux**: Any terminal (e.g., Ctrl + Alt + T).
2. Verify Git: `git --version`.
3. Clone the repository:
   ```
   git clone https://github.com/rhasspy/wyoming_streaming_asr.git
   ```
4. Navigate to the project folder:
   ```
   cd wyoming_streaming_asr
   ```

## _Preparing the Kroko Model_

### Prerequisites
- Project cloned (`wyoming_streaming_asr`) with `models/kroko` folder (may be empty).

### Step 1: Manually Download Model Files
1. Go to [https://huggingface.co/Banafo/Kroko-ASR/tree/main](https://huggingface.co/Banafo/Kroko-ASR/tree/main) in your browser.
2. Choose your language (e.g., German, English, Spanish—Kroko supports multiple languages).
3. Download data model file (64 or 128) into `wyoming_streaming_asr/models/kroko`:
   - Click file, hit "Download", and save to `models/kroko` in your project folder.
   - Ensure `enc.py` is in the project root or `models/kroko`.

### Step 2: Go to the directory and run enc.py
1. Run:
   - **Windows/Linux**:
     ```
     python enc.py  # Or python3 on Linux
     ```
2. This creates `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt` in `models/kroko`.

Done! Return to the main project directory and continue with the steps.

## Step 2: Create a Virtual Environment
1. Create the environment:
   - **Windows**: `python -m venv venv`
   - **Linux**: `python3 -m venv venv`
2. Activate the environment:
   - **Windows**: `venv\Scripts\activate`
   - **Linux**: `source venv/bin/activate`
   - You’ll see `(venv)` in the terminal.

## Step 3: Install Libraries
In the activated environment, run:
```
pip install sherpa-onnx wyoming numpy
```

## Step 4: Run the Server
In the activated environment, run:
- **Windows**:
  ```
  python -m wyoming_streaming_asr --model-dir "models\kroko" --language de
  ```
- **Linux**:
  ```
  python3 -m wyoming_streaming_asr --model-dir "models/kroko" --language de
  ```
- The server will start listening for audio.
- Stop it: Ctrl + C.

## Step 5: Deactivate
Deactivate the environment:
```
deactivate
```

## Tips
- Ensure `models/kroko` in the repository has the model files.
- For relaunch: activate `venv` and repeat Step 4.
