"""Transcribe Norwegian meeting recordings to text using faster-whisper.

The transcription runs entirely on local hardware, so recordings never leave
the machine. This avoids the data-processor and cross-border-transfer questions
that come with uploading meeting audio (which may contain personal data) to a
third-party cloud service.

Usage:
    python transcribe_meeting.py meeting.m4a
    python transcribe_meeting.py meeting.m4a --format both --output transcripts/
    python transcribe_meeting.py meeting.mp3 --model medium --device cpu \\
        --compute-type int8

Dependencies:
    pip install faster-whisper
    # ffmpeg must be available on PATH for non-wav input.

The first run downloads the chosen model weights (large-v3 is ~3 GB) and caches
them under ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class Segment(Protocol):
    """Minimal structural type for a transcription segment.

    faster-whisper's own ``Segment`` satisfies this, and tests can pass any
    lightweight object exposing the same three attributes.
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionConfig:
    """Runtime configuration for a transcription run.

    Attributes:
        model_size: Whisper model identifier, e.g. ``"large-v3"``, ``"medium"``.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        compute_type: ``"auto"``, ``"int8"`` (good for CPU), ``"float16"`` (GPU).
        language: ISO 639-1 code; ``"no"`` covers Norwegian bokmaal.
        beam_size: Beam width for decoding. Higher is slower but more accurate.
        vad_filter: Drop non-speech audio before decoding to reduce hallucination.
    """

    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "no"
    beam_size: int = 5
    vad_filter: bool = True


def format_timestamp(seconds: float, *, use_comma: bool = False) -> str:
    """Format a duration in seconds as ``HH:MM:SS.mmm``.

    Args:
        seconds: Non-negative duration in seconds.
        use_comma: Use a comma as the millisecond separator (SRT convention)
            instead of a period.

    Returns:
        A zero-padded timestamp string.

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    separator = "," if use_comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def _add_cuda_dll_directories() -> None:
    """Make pip-installed NVIDIA CUDA libraries loadable on Windows.

    ctranslate2 (faster-whisper's backend) loads cuBLAS and cuDNN at runtime
    when a CUDA device is used. When those libraries come from the
    ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels rather than a
    system-wide CUDA install, their DLLs live under ``site-packages/nvidia``
    and Windows will not find them unless the containing directories are
    registered on the DLL search path. This is a no-op on non-Windows
    platforms and when the wheels are not installed.
    """
    if sys.platform != 'win32':
        return

    import importlib.util
    import os

    for package in ('nvidia.cublas', 'nvidia.cudnn'):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue

        bin_dir = Path(spec.submodule_search_locations[0]) / 'bin'
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))


def segments_to_text(segments: Iterable[Segment]) -> str:
    """Join segment texts into a newline-separated plain-text transcript."""
    return "\n".join(segment.text.strip() for segment in segments)


def segments_to_srt(segments: Iterable[Segment]) -> str:
    """Render segments as an SRT subtitle document."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start, use_comma=True)
        end = format_timestamp(segment.end, use_comma=True)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def transcribe(
    audio_path: Path,
    config: TranscriptionConfig | None = None,
) -> tuple[list[Segment], object]:
    """Transcribe an audio file to a list of segments.

    The ``faster_whisper`` import is deferred to here so the formatting helpers
    above can be imported (and unit-tested) without the dependency installed.

    Args:
        audio_path: Path to an audio or video file readable by ffmpeg.
        config: Transcription settings; defaults to :class:`TranscriptionConfig`.

    Returns:
        A tuple of (materialized segment list, info object from the model).

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    _add_cuda_dll_directories()
    from faster_whisper import WhisperModel

    config = config or TranscriptionConfig()
    model = WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
    )
    segments, info = model.transcribe(
        str(audio_path),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
    )
    # Segments is a lazy generator; materialize it so callers can iterate twice
    # (e.g. to write both .txt and .srt).
    return list(segments), info
    # Diarization hook: to label speakers, run a pyannote.audio pipeline on the
    # same file and align its speaker turns against these segment timestamps.


def _write_outputs(
    segments: list[Segment],
    *,
    stem: str,
    output_dir: Path,
    fmt: str,
) -> list[Path]:
    """Write the requested transcript formats and return the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if fmt in {"txt", "both"}:
        txt_path = output_dir / f"{stem}.txt"
        txt_path.write_text(segments_to_text(segments), encoding="utf-8")
        written.append(txt_path)
    if fmt in {"srt", "both"}:
        srt_path = output_dir / f"{stem}.srt"
        srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
        written.append(srt_path)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a Norwegian meeting recording locally.",
    )
    parser.add_argument("audio", type=Path, help="Path to the audio/video file.")
    parser.add_argument("--model", default="large-v3", help="Whisper model size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", default="no", help="ISO 639-1 language code.")
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt", "both"],
        help="Output format(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (defaults to the input file's directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    config = TranscriptionConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    try:
        segments, info = transcribe(args.audio, config)
    except FileNotFoundError:
        print(f"error: file not found: {args.audio}", file=sys.stderr)
        return 1

    detected = getattr(info, "language", config.language)
    probability = getattr(info, "language_probability", None)
    if probability is not None:
        print(f"detected language: {detected} ({probability:.0%})", file=sys.stderr)

    output_dir = args.output or args.audio.parent
    written = _write_outputs(
        segments,
        stem=args.audio.stem,
        output_dir=output_dir,
        fmt=args.format,
    )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
