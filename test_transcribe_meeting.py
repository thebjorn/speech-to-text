"""Tests for transcribe_meeting's pure helpers.

These run without faster-whisper installed, since the model import is deferred
into transcribe(). A lightweight namedtuple stands in for a real Segment.

Run with: pytest test_transcribe_meeting.py
"""

from collections import namedtuple

import pytest

from transcribe_meeting import (
    SpeakerTurn,
    assign_speakers,
    format_timestamp,
    labeled_segments_to_srt,
    labeled_segments_to_text,
    segments_to_srt,
    segments_to_text,
)

FakeSegment = namedtuple("FakeSegment", ["start", "end", "text"])


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00.000"

    def test_sub_second_rounding(self):
        assert format_timestamp(1.2345) == "00:00:01.234"  # banker's rounding

    def test_hours_minutes_seconds(self):
        assert format_timestamp(3661.5) == "01:01:01.500"

    def test_srt_comma_separator(self):
        assert format_timestamp(5.25, use_comma=True) == "00:00:05.250".replace(
            ".", ","
        )

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_timestamp(-0.1)


class TestSegmentsToText:
    def test_strips_and_joins(self):
        segments = [
            FakeSegment(0.0, 1.0, "  God morgen.  "),
            FakeSegment(1.0, 2.0, "Skal vi begynne?"),
        ]
        assert segments_to_text(segments) == "God morgen.\nSkal vi begynne?"

    def test_empty(self):
        assert segments_to_text([]) == ""


class TestSegmentsToSrt:
    def test_single_block(self):
        segments = [FakeSegment(0.0, 2.5, "Velkommen.")]
        expected = "1\n00:00:00,000 --> 00:00:02,500\nVelkommen.\n"
        assert segments_to_srt(segments) == expected

    def test_block_numbering_and_separator(self):
        segments = [
            FakeSegment(0.0, 1.0, "Ett."),
            FakeSegment(1.0, 2.0, "To."),
        ]
        result = segments_to_srt(segments)
        assert result.startswith("1\n")
        assert "\n2\n" in result


class TestAssignSpeakers:
    def test_segment_within_single_turn(self):
        segments = [FakeSegment(1.0, 2.0, 'Hei.')]
        turns = [SpeakerTurn(0.0, 5.0, 'SPEAKER_00')]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == 'SPEAKER_00'

    def test_picks_speaker_with_most_overlap(self):
        segments = [FakeSegment(0.0, 10.0, 'Hei.')]
        turns = [
            SpeakerTurn(0.0, 3.0, 'SPEAKER_00'),
            SpeakerTurn(3.0, 10.0, 'SPEAKER_01'),
        ]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == 'SPEAKER_01'

    def test_sums_overlap_per_speaker(self):
        # SPEAKER_00 speaks 0-4 and 6-10 (8s total), SPEAKER_01 only 4-6 (2s).
        segments = [FakeSegment(0.0, 10.0, 'Hei.')]
        turns = [
            SpeakerTurn(0.0, 4.0, 'SPEAKER_00'),
            SpeakerTurn(4.0, 6.0, 'SPEAKER_01'),
            SpeakerTurn(6.0, 10.0, 'SPEAKER_00'),
        ]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == 'SPEAKER_00'

    def test_no_overlap_is_unknown(self):
        segments = [FakeSegment(20.0, 25.0, 'Hei.')]
        turns = [SpeakerTurn(0.0, 5.0, 'SPEAKER_00')]
        labeled = assign_speakers(segments, turns)
        assert labeled[0].speaker == 'UNKNOWN'

    def test_preserves_text_and_times(self):
        segments = [FakeSegment(1.0, 2.0, '  Hei.  ')]
        turns = [SpeakerTurn(0.0, 5.0, 'SPEAKER_00')]
        labeled = assign_speakers(segments, turns)
        assert (labeled[0].start, labeled[0].end) == (1.0, 2.0)
        assert labeled[0].text == '  Hei.  '

    def test_empty_segments(self):
        assert assign_speakers([], [SpeakerTurn(0.0, 1.0, 'SPEAKER_00')]) == []


class TestLabeledRenderers:
    def _labeled(self):
        segments = [
            FakeSegment(0.0, 1.0, '  God morgen.  '),
            FakeSegment(1.0, 2.0, 'Skal vi begynne?'),
        ]
        turns = [
            SpeakerTurn(0.0, 1.0, 'SPEAKER_00'),
            SpeakerTurn(1.0, 2.0, 'SPEAKER_01'),
        ]
        return assign_speakers(segments, turns)

    def test_labeled_text(self):
        result = labeled_segments_to_text(self._labeled())
        expected = 'SPEAKER_00: God morgen.\nSPEAKER_01: Skal vi begynne?'
        assert result == expected

    def test_labeled_srt(self):
        result = labeled_segments_to_srt(self._labeled())
        assert result.startswith(
            '1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: God morgen.\n'
        )
        assert '\n2\n00:00:01,000 --> 00:00:02,000\nSPEAKER_01:' in result
