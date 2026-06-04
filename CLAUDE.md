# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A privacy-first CLI that transcribes meeting/interview recordings to text **entirely on local hardware** (recordings never leave the machine), with optional speaker diarization. Built on `faster-whisper` (transcription) and `pyannote.audio` (diarization). Default decoding language is Norwegian (`no`). The whole tool is one module — `transcribe_meeting.py` — plus its test file. Developed on Windows; commands below use the project venv directly.

## Commands

```powershell
# Environment (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # faster-whisper + pytest

# Tests — run WITHOUT faster-whisper/torch/pyannote needing to work (see below)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest test_transcribe_meeting.py::TestAssignSpeakers::test_no_overlap_is_unknown   # single test

# Run the CLI (or `transcribe-meeting <audio>` after `pip install -e .`)
.\.venv\Scripts\python.exe transcribe_meeting.py meeting.m4a --diarize --device cuda --timestamps --format both --output transcripts
.\.venv\Scripts\python.exe transcribe_meeting.py --prime --model large-v3 --diarize   # pre-download model weights, no audio
```

There is no build step, linter, or CI configured. Style conventions: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, 4-space indent, lines ≤79 (up to 100 when needed). Strings use **single quotes** — keep double only when a string contains an apostrophe; docstrings use triple double-quotes (`"""`).

## Architecture

**Two-tier module design — preserve it.** `transcribe_meeting.py` splits into:

- **Pure, import-light helpers** (timestamp/`_clock`, `segments_to_text`/`segments_to_srt`, `assign_speakers`/`_best_speaker`, `merge_consecutive_speakers`, `labeled_segments_to_*`, `_choose_diarization_device`, `_load_dotenv`). These have no heavy dependencies and are the only things unit-tested.
- **Heavy functions** (`transcribe`, `diarize`, `prime`, `_build_diarization_pipeline`) that **defer** `import faster_whisper` / `import torch` / `from pyannote.audio import Pipeline` to *inside the function body*. This is why the test suite runs with only `pytest` installed.

When adding behavior, put the logic in a pure helper and unit-test it; only the actual model calls belong in the deferred-import functions. `Segment` is a `Protocol` (start/end/text) so tests pass plain namedtuples where the real faster-whisper `Segment` would go.

**Two independent ML stacks.** Transcription uses faster-whisper → ctranslate2 (+ NVIDIA cuBLAS/cuDNN/cudart wheels). Diarization uses pyannote → torch. They share nothing: GPU transcription works regardless of the torch situation. `main()` calls `transcribe()` first, then `diarize()` only if `--diarize`. `--device` (auto/cpu/cuda) feeds both.

**Diarization decodes via PyAV, not pyannote's own loader.** `diarize()` decodes audio with `faster_whisper.audio.decode_audio` (PyAV, format-agnostic, bundled) and hands pyannote a `{'waveform', 'sample_rate'}` dict. This deliberately bypasses torchcodec's file decoding, which returns slightly-off sample counts for MP3 and trips pyannote's strict chunk extraction. Don't switch `diarize()` back to passing a file path.

**Output shapes** (`_write_outputs`): `.txt` merges consecutive same-speaker turns into per-speaker paragraphs (`merge_consecutive_speakers`); `.srt` stays one cue per segment (subtitles need fine timing). `--timestamps` adds `[HH:MM:SS]` prefixes to `.txt` only. `LabeledSegment` carries raw text + speaker; the renderers add the `SPEAKER:` prefix.

## Windows / GPU gotchas (these caused real, multi-step debugging)

- **ctranslate2 finds CUDA libs via PATH, not just `add_dll_directory`.** `_add_cuda_dll_directories()` registers the `nvidia-*-cu12` wheel `bin` dirs (cublas, **cuda_runtime/cudart**, nvrtc, cudnn) with `os.add_dll_directory` *and* prepends them to `PATH`. cuBLAS needs cudart, so `nvidia-cuda-runtime-cu12` must be installed (it's in `requirements-gpu.txt`).
- **GPU diarization version matrix.** pyannote 4.x needs torch ≥ 2.12 (via torchcodec 0.14). That torch ships on the **cu126** index, **not cu124** (cu124 caps at torch 2.6, which breaks the stack). So a CUDA build must come from cu126 *and* the NVIDIA driver must support CUDA 12.6+:
  `pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126`
- **Diarization needs a full-shared ffmpeg build** (ships `avcodec-*.dll` etc.) for torchcodec to import. Point `FFMPEG_BIN` at its `bin` folder (e.g. in `.env`); `_add_ffmpeg_dll_directory()` registers it. The static "essentials" build won't work.

## Config & secrets

`main()` calls `_load_dotenv()` which reads `.env` from the **current working directory** — run the CLI from the project dir so it's picked up. Keys: `HF_TOKEN` (Hugging Face token for diarization), `FFMPEG_BIN`. Precedence for the token: `--hf-token` flag → shell env → `.env`. `.env` is gitignored; `.env.example` documents the keys.

The pyannote model is gated: a Hugging Face token (read scope) must have accepted terms for **both** `pyannote/speaker-diarization-3.1` **and** its dependency `pyannote/segmentation-3.0`.

`.venv/`, `.env`, audio media, and `transcripts/`/`*.srt` are gitignored — regenerating outputs into `transcripts/` is normal and won't be committed.

## Line endings

`.gitattributes` normalizes line endings to **LF** (`* text=auto eol=lf`, with `eol=crlf` for `*.bat`/`*.cmd`/`*.ps1` and `binary` for audio formats). This intentionally overrides a global `core.autocrlf=true`, so expect LF in the working tree and don't "fix" it — without this, Git warns about LF↔CRLF conversion on every commit on Windows.
