"""Comparing two screenshots of the same step.

The number this produces drives which step the UI opens on, so the properties
that matter are about *not* crying wolf: identical pages read as identical,
trivial rendering noise does not, and anything unreadable produces no
comparison at all rather than a misleading zero.
"""

from __future__ import annotations

import io

import pytest

from imagediff import describe, ratio

PIL = pytest.importorskip("PIL")


def png(colour, size=(64, 48), patch=None) -> bytes:
    """A solid image, optionally with a rectangle painted on it."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, colour)
    if patch:
        box, patch_colour = patch
        ImageDraw.Draw(image).rectangle(box, fill=patch_colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_same_page_reads_as_identical():
    assert ratio(png("white"), png("white")) == 0.0


def test_a_completely_different_page_reads_as_completely_different():
    assert ratio(png("white"), png("black")) == 1.0


def test_a_small_change_is_a_small_number():
    """A quarter of the page changing should read as roughly a quarter."""
    changed = png("white", patch=((0, 0, 31, 23), "black"))
    value = ratio(png("white"), changed)
    assert 0.2 < value < 0.3


def test_rendering_noise_does_not_register():
    """A pixel that moved by one or two values is anti-aliasing, not a change.

    Without a tolerance every comparison would have a floor above zero and
    "identical" would never be reported for a real page.
    """
    assert ratio(png((255, 255, 255)), png((250, 250, 250))) == 0.0


def test_a_difference_in_one_channel_is_not_diluted_by_two_that_held_still():
    """Per pixel the largest channel change wins. Averaging would let a strong
    change in one colour disappear."""
    assert ratio(png((0, 0, 0)), png((0, 0, 255))) == 1.0


def test_images_of_different_sizes_are_still_comparable():
    """A viewport that changed size must not make every step incomparable."""
    assert ratio(png("white", size=(64, 48)), png("white", size=(128, 96))) == 0.0


# --- refusing to answer ----------------------------------------------------


@pytest.mark.parametrize(
    "before,after",
    [(None, b"x"), (b"x", None), (None, None), (b"", b"")],
)
def test_a_missing_image_is_no_comparison_rather_than_zero(before, after):
    """`None` and `0.0` mean different things.

    A UI that conflated them would report a first run -- which has no baseline
    at all -- as a perfect match with something that does not exist.
    """
    assert ratio(before, after) is None


def test_bytes_that_are_not_an_image_produce_no_comparison():
    """A diff is a diagnostic about work that already happened. It must never
    raise into the run it describes."""
    assert ratio(b"not a png", b"also not a png") is None


# --- what a person reads ---------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "no baseline"),
        (0.0, "identical"),
        (0.004, "almost identical"),
        (0.05, "small differences"),
        (0.25, "noticeably different"),
        (0.9, "very different"),
    ],
)
def test_the_ratio_is_described_in_words(value, expected):
    """Four decimal places is not a thing to put in a timeline."""
    assert describe(value) == expected
