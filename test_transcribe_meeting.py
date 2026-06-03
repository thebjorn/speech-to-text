"""Tests for transcribe_meeting's pure helpers.

These run without faster-whisper installed, since the model import is deferred
into transcribe(). A lightweight namedtuple stands in for a real Segment.

Run with: pytest test_transcribe_meeting.py
"""

from collections import namedtuple

import pytest

from transcribe_meeting import (
    format_timestamp,
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
