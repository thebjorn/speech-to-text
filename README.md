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

Speaker labels (diarization) are an optional extra with their own setup — see
[Speaker labels](#speaker-labels-diarization) below.

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

# Pre-download/cache the models ahead of time (no audio needed)
python transcribe_meeting.py --prime --model large-v3
python transcribe_meeting.py --prime --model large-v3 --diarize
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
| `--diarize`      | off         | Label each segment by speaker (see below).        |
| `--hf-token`     | `$HF_TOKEN` | Hugging Face token for the diarization model.     |
| `--speakers`     | auto        | Exact number of speakers, if known.               |
| `--prime`        | off         | Download/cache the models, then exit (no audio).  |
| `--timestamps`   | off         | Prefix each `.txt` line with a `[HH:MM:SS]` time.  |

### Recommended for your hardware

On an NVIDIA RTX 4000 Ada (20 GB VRAM), the highest-quality model runs
comfortably:

```powershell
python transcribe_meeting.py meeting.m4a --model large-v3 --device cuda --compute-type float16
```

## Speaker labels (diarization)

With `--diarize`, the transcript is annotated with who spoke each line, using a
[pyannote.audio](https://github.com/pyannote/pyannote-audio) pipeline that
detects speaker turns. Each transcription segment is attributed to the speaker
whose turns overlap it the most, and consecutive turns by the same speaker are
merged into one paragraph in the `.txt`. With `--timestamps`, each paragraph is
prefixed with its `[HH:MM:SS]` start time:

```
[00:00:00] SPEAKER_00: God morgen, skal vi begynne?
[00:00:04] SPEAKER_01: Ja, la oss starte med budsjettet. Først tallene fra i fjor.
```

(The `.srt` keeps one cue per segment — subtitles need fine-grained timing — so
merging applies only to the `.txt`.)

This is an opt-in feature with extra setup, because the model is large and
gated:

1. **Install the extra** (pulls in PyTorch — see `requirements-diarize.txt` for
   the GPU/CUDA torch note):
   ```powershell
   pip install -r requirements-diarize.txt
   # or: pip install -e ".[diarize]"
   ```
2. **Accept the model terms** once at
   <https://hf.co/pyannote/speaker-diarization-3.1> (and the segmentation model
   it depends on, linked from that page).
3. **Provide a Hugging Face token** (needs only **read** access) — create one
   at <https://hf.co/settings/tokens>. Supply it in any of these ways:
   - a `.env` file in the project directory (copy `.env.example` → `.env`; it's
     gitignored, so the token stays out of the repo):
     ```
     HF_TOKEN=hf_...
     ```
   - a shell environment variable: `$env:HF_TOKEN = "hf_..."`
   - the `--hf-token hf_...` flag

   Then:
   ```powershell
   python transcribe_meeting.py meeting.m4a --diarize --format both
   ```

   Precedence is flag → shell env var → `.env` file.

Diarization honors `--device` (shared with transcription): `auto` (default)
uses the GPU when a CUDA-enabled PyTorch is installed and falls back to CPU
otherwise, `cuda` forces the GPU (and errors clearly if your torch has no CUDA
support), and `cpu` forces CPU. The device it picked is printed to stderr. The
default `pip install` of torch on Windows is CPU-only — install a CUDA build for
GPU diarization (see `requirements-diarize.txt`):

```powershell
# pyannote 4.x needs torch >= 2.12, which is on the cu126 index (NOT cu124 --
# that tops out at torch 2.6, too old). Your NVIDIA driver must support
# CUDA 12.6+ (check nvidia-smi).
pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python transcribe_meeting.py meeting.m4a --diarize --device cuda
```

If you know the number of participants, `--speakers N` improves accuracy.

> **ffmpeg note:** pyannote decodes audio via `torchcodec`, which needs ffmpeg's
> shared libraries — install the **full-shared** ffmpeg build (the one that
> ships `avcodec-*.dll` etc.), not the static "essentials" build. Either put its
> `bin` folder on your `PATH`, or set `FFMPEG_BIN` to that folder (e.g. in
> `.env`) and the program registers the DLLs for you. Without it you'll see a
> `libtorchcodec` load error. (Plain transcription doesn't need this — only
> diarization does.) To download the models without decoding audio, use
> `--prime`.

## Development

```powershell
pip install -r requirements-dev.txt
pytest
```

The test suite (`test_transcribe_meeting.py`) covers the pure helpers — timestamp
formatting, transcript rendering, and speaker-to-segment alignment — and runs
**without** faster-whisper, pyannote, or torch installed. The heavy model
imports are deferred into `transcribe()` and `diarize()`, so the logic stays
unit-testable.
