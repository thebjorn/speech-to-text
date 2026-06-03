# speech-to-text

Transcribe meeting recordings to text **entirely on local hardware** using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Because transcription runs on your machine, recordings never leave it. This
sidesteps the data-processor and cross-border-transfer questions that come with
uploading meeting audio (which may contain personal data) to a third-party
cloud service.

Default output is a plain-text transcript; SRT subtitles are also supported.
The default decoding language is Norwegian (`no`), configurable per run.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (needed for any non-WAV input,
  e.g. `.m4a` / `.mp3` / `.mp4`)
- Optional but recommended: an NVIDIA GPU for fast transcription

## Install

From the project directory:

```powershell
# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2a. CPU-only install
pip install -r requirements-dev.txt

# 2b. ...or install with GPU acceleration (NVIDIA, CUDA 12 + cuDNN 9)
pip install -r requirements-dev.txt -r requirements-gpu.txt
```

Alternatively, using the project metadata in `pyproject.toml`:

```powershell
pip install -e ".[dev,gpu]"
```

The first transcription run downloads the chosen model weights (`large-v3` is
~3 GB) and caches them under `~/.cache/huggingface`.

## GPU acceleration

faster-whisper uses CUDA through ctranslate2 when a CUDA device is available.
The `requirements-gpu.txt` extras install the cuBLAS and cuDNN runtime libraries
as wheels, so you don't need a system-wide CUDA Toolkit install. On Windows,
`transcribe_meeting.py` registers those wheels' DLL directories automatically at
import time, so GPU support works out of the box once the extras are installed.

Verify the GPU is visible to ctranslate2:

```powershell
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), 'CUDA device(s)')"
```

With device set to `auto` (the default), a CUDA device is used when present and
the code falls back to CPU otherwise.

## Usage

```powershell
# Plain-text transcript next to the input file
python transcribe_meeting.py meeting.m4a

# Both .txt and .srt, written to a transcripts/ directory
python transcribe_meeting.py meeting.m4a --format both --output transcripts/

# Force CPU with an int8 model (smaller, faster on CPU)
python transcribe_meeting.py meeting.mp3 --model medium --device cpu --compute-type int8
```

If installed with `pip install -e .`, a `transcribe-meeting` console command is
also available with the same arguments.

### Options

| Flag             | Default     | Description                                       |
| ---------------- | ----------- | ------------------------------------------------- |
| `--model`        | `large-v3`  | Whisper model size (`tiny`…`large-v3`).           |
| `--device`       | `auto`      | `auto`, `cpu`, or `cuda`.                         |
| `--compute-type` | `auto`      | e.g. `float16` (GPU), `int8` (CPU).               |
| `--language`     | `no`        | ISO 639-1 language code (`no` = Norwegian bokmål).|
| `--format`       | `txt`       | `txt`, `srt`, or `both`.                          |
| `--output`       | input's dir | Output directory.                                 |

### Recommended for your hardware

On an NVIDIA RTX 4000 Ada (20 GB VRAM), the highest-quality model runs
comfortably:

```powershell
python transcribe_meeting.py meeting.m4a --model large-v3 --device cuda --compute-type float16
```

## Development

```powershell
pip install -r requirements-dev.txt
pytest
```

The test suite (`test_transcribe_meeting.py`) covers the pure formatting helpers
and runs **without** faster-whisper installed — the heavy model import is
deferred into `transcribe()`.

## Roadmap / ideas

- **Speaker labels (diarization):** run a `pyannote.audio` pipeline on the same
  file and align speaker turns against the segment timestamps. A hook for this
  is noted in `transcribe()`.
