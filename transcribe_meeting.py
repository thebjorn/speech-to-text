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
import os
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


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous span of audio attributed to one speaker by diarization.

    Attributes:
        start: Turn start time in seconds.
        end: Turn end time in seconds.
        speaker: Diarization label, e.g. ``"SPEAKER_00"``.
    """

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class LabeledSegment:
    """A transcription segment annotated with a speaker label.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        text: The transcribed text (unmodified).
        speaker: The speaker label assigned by :func:`assign_speakers`.
    """

    start: float
    end: float
    text: str
    speaker: str


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


def _best_speaker(segment: Segment, turns: Iterable[SpeakerTurn]) -> str:
    """Return the speaker whose turns overlap ``segment`` the most.

    Overlap is summed across all turns for each speaker, so a segment that
    straddles two short turns by the same speaker is attributed correctly.
    Returns ``"UNKNOWN"`` when no turn overlaps the segment.
    """
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > 0:
            overlap_by_speaker[turn.speaker] = (
                overlap_by_speaker.get(turn.speaker, 0.0) + overlap
            )

    if not overlap_by_speaker:
        return 'UNKNOWN'

    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def assign_speakers(
    segments: Iterable[Segment],
    turns: Iterable[SpeakerTurn],
) -> list[LabeledSegment]:
    """Attach a speaker label to each transcription segment.

    Pure function: it only needs the start/end times of ``segments`` and
    ``turns``, so it is unit-testable without faster-whisper or pyannote.

    Args:
        segments: Transcription segments, in time order.
        turns: Diarization turns from :func:`diarize`.

    Returns:
        One :class:`LabeledSegment` per input segment, preserving order.
    """
    turns = list(turns)
    return [
        LabeledSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=_best_speaker(segment, turns),
        )
        for segment in segments
    ]


def labeled_segments_to_text(segments: Iterable[LabeledSegment]) -> str:
    """Render labeled segments as ``SPEAKER: text`` lines."""
    return '\n'.join(
        f'{segment.speaker}: {segment.text.strip()}' for segment in segments
    )


def labeled_segments_to_srt(segments: Iterable[LabeledSegment]) -> str:
    """Render labeled segments as SRT, prefixing each cue with the speaker."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start, use_comma=True)
        end = format_timestamp(segment.end, use_comma=True)
        text = f'{segment.speaker}: {segment.text.strip()}'
        blocks.append(f'{index}\n{start} --> {end}\n{text}\n')
    return '\n'.join(blocks)


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


def diarize(
    audio_path: Path,
    *,
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Detect who-spoke-when with a pyannote.audio diarization pipeline.

    Like :func:`transcribe`, the heavy import is deferred so the rest of the
    module (and its tests) work without pyannote or torch installed.

    The ``pyannote/speaker-diarization-3.1`` model is gated on the Hugging Face
    Hub: you must accept its conditions once at
    https://hf.co/pyannote/speaker-diarization-3.1 and supply an access token,
    either via ``hf_token`` or the ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN``
    environment variable.

    Args:
        audio_path: Path to an audio or video file readable by the pipeline.
        hf_token: Hugging Face access token; falls back to the environment.
        num_speakers: Exact number of speakers, if known. ``None`` lets the
            pipeline estimate it.

    Returns:
        Speaker turns sorted by start time.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
        RuntimeError: If no Hugging Face token can be found.
    """
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    token = (
        hf_token
        or os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGINGFACE_TOKEN')
    )
    if not token:
        raise RuntimeError(
            'speaker diarization needs a Hugging Face access token; set '
            'HF_TOKEN or pass --hf-token, and accept the model terms at '
            'https://hf.co/pyannote/speaker-diarization-3.1'
        )

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1',
        use_auth_token=token,
    )

    # Diarization is also much faster on the GPU when one is available.
    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda'))

    annotation = pipeline(str(audio_path), num_speakers=num_speakers)
    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: turn.start)
    return turns


def _write_outputs(
    segments: list[Segment],
    *,
    stem: str,
    output_dir: Path,
    fmt: str,
    diarized: bool = False,
) -> list[Path]:
    """Write the requested transcript formats and return the paths written.

    When ``diarized`` is true, ``segments`` are :class:`LabeledSegment` values
    and the speaker-aware renderers are used instead of the plain ones.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    to_text = labeled_segments_to_text if diarized else segments_to_text
    to_srt = labeled_segments_to_srt if diarized else segments_to_srt

    written: list[Path] = []
    if fmt in {"txt", "both"}:
        txt_path = output_dir / f"{stem}.txt"
        txt_path.write_text(to_text(segments), encoding="utf-8")
        written.append(txt_path)
    if fmt in {"srt", "both"}:
        srt_path = output_dir / f"{stem}.srt"
        srt_path.write_text(to_srt(segments), encoding="utf-8")
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
    parser.add_argument(
        '--diarize',
        action='store_true',
        help='Label each segment by speaker using pyannote.audio.',
    )
    parser.add_argument(
        '--hf-token',
        default=None,
        help='Hugging Face token for the diarization model (or set HF_TOKEN).',
    )
    parser.add_argument(
        '--speakers',
        type=int,
        default=None,
        help='Exact number of speakers, if known (default: auto-detect).',
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

    if args.diarize:
        try:
            turns = diarize(
                args.audio,
                hf_token=args.hf_token,
                num_speakers=args.speakers,
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        speakers = {turn.speaker for turn in turns}
        print(f"diarized {len(turns)} turns, {len(speakers)} speakers",
              file=sys.stderr)
        segments = assign_speakers(segments, turns)

    written = _write_outputs(
        segments,
        stem=args.audio.stem,
        output_dir=output_dir,
        fmt=args.format,
        diarized=args.diarize,
    )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
