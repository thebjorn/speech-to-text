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

    model_size: str = 'large-v3'
    device: str = 'auto'
    compute_type: str = 'auto'
    language: str = 'no'
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
        raise ValueError('seconds must be non-negative')
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    separator = ',' if use_comma else '.'
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}'


def _clock(seconds: float) -> str:
    """Format ``seconds`` as a bare ``HH:MM:SS`` clock (no milliseconds).
    """
    return format_timestamp(seconds)[:8]


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

    # ctranslate2 loads cuBLAS at runtime, which in turn needs the CUDA runtime
    # (cudart) and nvrtc. Register each wheel's bin directory so the chain
    # resolves without a system-wide CUDA install.
    for package in (
        'nvidia.cublas',
        'nvidia.cuda_runtime',
        'nvidia.cuda_nvrtc',
        'nvidia.cudnn',
    ):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue

        bin_dir = Path(spec.submodule_search_locations[0]) / 'bin'
        if bin_dir.is_dir():
            # add_dll_directory covers ctypes-based loads; ctranslate2's own
            # loader searches PATH, so prepend the directory there too.
            os.add_dll_directory(str(bin_dir))
            os.environ['PATH'] = (
                str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
            )


def _add_ffmpeg_dll_directory() -> None:
    """Make a shared-build FFmpeg discoverable for pyannote's torchcodec backend.

       pyannote.audio decodes audio through torchcodec, which loads FFmpeg's shared
       DLLs (``avcodec``/``avformat``/``avutil``/...) at runtime. When FFmpeg is not
       already on ``PATH``, set the ``FFMPEG_BIN`` environment variable (e.g. in
       ``.env``) to its ``bin`` directory and this registers it on the DLL search
       path. No-op on non-Windows platforms and when ``FFMPEG_BIN`` is unset or does
       not point at a directory.
    """
    if sys.platform != 'win32':
        return

    ffmpeg_bin = os.environ.get('FFMPEG_BIN')
    if not ffmpeg_bin or not Path(ffmpeg_bin).is_dir():
        return

    os.add_dll_directory(ffmpeg_bin)
    os.environ['PATH'] = ffmpeg_bin + os.pathsep + os.environ.get('PATH', '')


def segments_to_text(
    segments: Iterable[Segment],
    *,
    timestamps: bool = False,
) -> str:
    """Join segment texts into a newline-separated plain-text transcript.

       With ``timestamps``, each line is prefixed with a ``[HH:MM:SS]`` start time.
    """
    lines = []
    for segment in segments:
        prefix = f'[{_clock(segment.start)}] ' if timestamps else ''
        lines.append(f'{prefix}{segment.text.strip()}')
    return '\n'.join(lines)


def segments_to_srt(segments: Iterable[Segment]) -> str:
    """Render segments as an SRT subtitle document.
    """
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start, use_comma=True)
        end = format_timestamp(segment.end, use_comma=True)
        blocks.append(f'{index}\n{start} --> {end}\n{segment.text.strip()}\n')
    return '\n'.join(blocks)


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


def merge_consecutive_speakers(
    segments: Iterable[LabeledSegment],
) -> list[LabeledSegment]:
    """Merge consecutive same-speaker labeled segments into one block each.

       Pure. Each returned :class:`LabeledSegment` spans one speaker's run of
       consecutive turns: ``start``/``end`` cover the run and ``text`` is the
       turns joined with a space. This turns one-line-per-segment output into
       readable per-speaker paragraphs.
    """
    merged: list[LabeledSegment] = []
    for segment in segments:
        if merged and merged[-1].speaker == segment.speaker:
            previous = merged[-1]
            merged[-1] = LabeledSegment(
                start=previous.start,
                end=segment.end,
                text=f'{previous.text} {segment.text.strip()}',
                speaker=previous.speaker,
            )
        else:
            merged.append(
                LabeledSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    speaker=segment.speaker,
                )
            )
    return merged


def labeled_segments_to_text(
    segments: Iterable[LabeledSegment],
    *,
    timestamps: bool = False,
) -> str:
    """Render labeled segments as ``SPEAKER: text`` lines.

       With ``timestamps``, each line is prefixed with a ``[HH:MM:SS]`` start time.
    """
    lines = []
    for segment in segments:
        prefix = f'[{_clock(segment.start)}] ' if timestamps else ''
        lines.append(f'{prefix}{segment.speaker}: {segment.text.strip()}')
    return '\n'.join(lines)


def labeled_segments_to_srt(segments: Iterable[LabeledSegment]) -> str:
    """Render labeled segments as SRT, prefixing each cue with the speaker.
    """
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


DIARIZATION_MODEL = 'pyannote/speaker-diarization-3.1'


def _choose_diarization_device(requested: str, cuda_available: bool) -> str:
    """Resolve the diarization torch device from the ``--device`` choice.

       Pure (imports no torch) so it stays unit-testable. Returns ``'cuda'`` or
       ``'cpu'``.

       Args:
           requested: ``'auto'``, ``'cpu'``, or ``'cuda'``.
           cuda_available: Whether the installed torch reports CUDA support.

       Raises:
           RuntimeError: If ``'cuda'`` is requested but not available.
    """
    if requested == 'cpu':
        return 'cpu'

    if requested == 'cuda':
        if not cuda_available:
            raise RuntimeError(
                'diarization requested --device cuda, but this PyTorch build '
                'has no CUDA support. Install a CUDA build, e.g.: '
                'pip install torch --index-url '
                'https://download.pytorch.org/whl/cu124'
            )
        return 'cuda'

    # 'auto': prefer the GPU when torch reports CUDA support.
    return 'cuda' if cuda_available else 'cpu'


def _build_diarization_pipeline(
    hf_token: str | None = None,
    device: str = 'auto',
):
    """Load the pyannote diarization pipeline on the chosen device.

       Resolves the Hugging Face token from ``hf_token`` or the environment, and
       raises a clear error if none is set. ``device`` is ``'auto'`` (GPU when a
       CUDA torch is available, else CPU), ``'cpu'``, or ``'cuda'``. Factored out
       so :func:`diarize` and :func:`prime` construct (and thus download) the
       pipeline the same way.
    """
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

    # pyannote decodes audio via torchcodec, which needs FFmpeg's shared DLLs.
    _add_ffmpeg_dll_directory()

    import torch
    from pyannote.audio import Pipeline

    # The auth kwarg was renamed from use_auth_token to token in newer
    # pyannote.audio / huggingface_hub releases; support both.
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=token
        )

    resolved = _choose_diarization_device(device, torch.cuda.is_available())
    print(f'diarization device: {resolved}', file=sys.stderr)
    if resolved == 'cpu':
        print('  (install a CUDA build of torch for faster diarization)',
              file=sys.stderr)
    pipeline.to(torch.device(resolved))
    return pipeline


def diarize(
    audio_path: Path,
    *,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    device: str = 'auto',
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
           device: ``'auto'`` (GPU when a CUDA torch is available, else CPU),
               ``'cpu'``, or ``'cuda'``.

       Returns:
           Speaker turns sorted by start time.

       Raises:
           FileNotFoundError: If ``audio_path`` does not exist.
           RuntimeError: If no Hugging Face token can be found, or ``device`` is
               ``'cuda'`` but the installed torch has no CUDA support.
    """
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    pipeline = _build_diarization_pipeline(hf_token, device=device)

    # Decode to a 16 kHz mono waveform with PyAV (faster-whisper's decoder) and
    # hand it to pyannote directly, rather than letting pyannote open the file.
    # PyAV decodes every container consistently (mp3/m4a/opus/flac/wav), and
    # this avoids torchcodec quirks -- notably MP3 decoding returning a slightly
    # off sample count, which trips pyannote's strict chunk extraction.
    import torch
    from faster_whisper.audio import decode_audio

    waveform = torch.from_numpy(decode_audio(str(audio_path))).unsqueeze(0)
    result = pipeline(
        {'waveform': waveform, 'sample_rate': 16000},
        num_speakers=num_speakers,
    )

    # pyannote 4.x returns a DiarizeOutput wrapping the annotation; 3.x returns
    # the Annotation directly. Both expose itertracks().
    annotation = getattr(result, 'speaker_diarization', result)
    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: turn.start)
    return turns


def prime(
    config: TranscriptionConfig | None = None,
    *,
    diarize: bool = False,
    hf_token: str | None = None,
) -> None:
    """Download and cache the model files so the first real run is fast.

       Loads the configured Whisper model (which downloads its weights on first
       use) and, when ``diarize`` is true, the pyannote diarization pipeline.
       Nothing is transcribed; this only warms the on-disk model cache.
    """
    config = config or TranscriptionConfig()

    print(f"priming whisper model '{config.model_size}'...", file=sys.stderr)
    _add_cuda_dll_directories()
    from faster_whisper import WhisperModel

    WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
    )

    if diarize:
        print('priming diarization pipeline...', file=sys.stderr)
        _build_diarization_pipeline(hf_token, device=config.device)


def _write_outputs(
    segments: list[Segment],
    *,
    stem: str,
    output_dir: Path,
    fmt: str,
    diarized: bool = False,
    timestamps: bool = False,
) -> list[Path]:
    """Write the requested transcript formats and return the paths written.

       When ``diarized`` is true, ``segments`` are :class:`LabeledSegment` values:
       the ``.txt`` merges consecutive same-speaker turns into per-speaker
       paragraphs, while the ``.srt`` keeps one cue per segment (subtitles need
       fine-grained timing). ``timestamps`` prefixes each ``.txt`` line with a
       ``[HH:MM:SS]`` start time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if fmt in {'txt', 'both'}:
        if diarized:
            text = labeled_segments_to_text(
                merge_consecutive_speakers(segments), timestamps=timestamps
            )
        else:
            text = segments_to_text(segments, timestamps=timestamps)
        txt_path = output_dir / f'{stem}.txt'
        txt_path.write_text(text, encoding='utf-8')
        written.append(txt_path)
    if fmt in {'srt', 'both'}:
        if diarized:
            srt = labeled_segments_to_srt(segments)
        else:
            srt = segments_to_srt(segments)
        srt_path = output_dir / f'{stem}.srt'
        srt_path.write_text(srt, encoding='utf-8')
        written.append(srt_path)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Transcribe a Norwegian meeting recording locally.',
    )
    parser.add_argument(
        'audio',
        type=Path,
        nargs='?',
        default=None,
        help='Path to the audio/video file (omit when using --prime).',
    )
    parser.add_argument('--model', default='large-v3', help='Whisper model size.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--compute-type', default='auto')
    parser.add_argument('--language', default='no', help='ISO 639-1 language code.')
    parser.add_argument(
        '--format',
        default='txt',
        choices=['txt', 'srt', 'both'],
        help='Output format(s).',
    )
    parser.add_argument(
        '--output',
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
    parser.add_argument(
        '--prime',
        action='store_true',
        help='Download and cache model files, then exit (no transcription).',
    )
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='Prefix each transcript line with a [HH:MM:SS] start time.',
    )
    return parser.parse_args(argv)


def _load_dotenv(path: Path = Path('.env')) -> None:
    """Populate ``os.environ`` from a ``.env`` file, if one is present.

       Lines are ``KEY=value`` (an optional ``export`` prefix is allowed); blank
       lines and ``#`` comments are skipped, and surrounding quotes are stripped.
       Existing environment variables win, so a value exported in the shell takes
       precedence over the file. This avoids a python-dotenv dependency for the
       one common case: keeping HF_TOKEN out of your shell history and the repo.
    """
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()

        key, separator, value = line.partition('=')
        if not separator:
            continue

        key = key.strip()
        value = value.strip().strip('\'"')
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.
    """
    _load_dotenv()
    args = _parse_args(argv)
    config = TranscriptionConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    if args.prime:
        try:
            prime(config, diarize=args.diarize, hf_token=args.hf_token)
        except RuntimeError as error:
            print(f'error: {error}', file=sys.stderr)
            return 1
        print('primed: model files are downloaded and cached')
        return 0

    if args.audio is None:
        print('error: an audio file is required (or use --prime)',
              file=sys.stderr)
        return 2

    try:
        segments, info = transcribe(args.audio, config)
    except FileNotFoundError:
        print(f'error: file not found: {args.audio}', file=sys.stderr)
        return 1

    detected = getattr(info, 'language', config.language)
    probability = getattr(info, 'language_probability', None)
    if probability is not None:
        print(f'detected language: {detected} ({probability:.0%})', file=sys.stderr)

    output_dir = args.output or args.audio.parent

    if args.diarize:
        try:
            turns = diarize(
                args.audio,
                hf_token=args.hf_token,
                num_speakers=args.speakers,
                device=args.device,
            )
        except RuntimeError as error:
            print(f'error: {error}', file=sys.stderr)
            return 1

        speakers = {turn.speaker for turn in turns}
        print(f'diarized {len(turns)} turns, {len(speakers)} speakers',
              file=sys.stderr)
        segments = assign_speakers(segments, turns)

    written = _write_outputs(
        segments,
        stem=args.audio.stem,
        output_dir=output_dir,
        fmt=args.format,
        diarized=args.diarize,
        timestamps=args.timestamps,
    )
    for path in written:
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
