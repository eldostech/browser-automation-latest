"""How much two screenshots differ.

One number per step, computed when the step is recorded rather than when the
timeline is drawn: a batch of a thousand rows would otherwise decode two images
per step every time somebody opened the page.

**A ratio, not a verdict.** This says what fraction of pixels changed; it does
not say whether that matters. A rendered timestamp changes a handful of pixels
on every run and means nothing; a form that failed to submit can change very
few pixels and mean everything. The UI sorts by this and shows the first
divergence, and a person decides.

**Pillow, not a similarity metric.** SSIM and friends want numpy and model
perceptual similarity, which is a better answer to a question nobody here is
asking. What this needs is "did the page change, roughly how much, and where do
I look first" -- and a downscaled per-pixel comparison answers that in a few
milliseconds.

Nothing here may raise into a run. A diff is a diagnostic about work that has
already happened, so every failure returns ``None`` and the step is recorded
without one.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

#: Both images are scaled to this before comparing. It makes the cost
#: independent of viewport size, and it blurs away the single-pixel noise of
#: font rendering and cursor position that would otherwise put a floor under
#: every comparison.
COMPARE_WIDTH = 320
COMPARE_HEIGHT = 240

#: A channel difference below this is not a change. JPEG-ish artefacts and
#: anti-aliasing move a pixel by one or two values without anything happening.
CHANNEL_TOLERANCE = 16


def ratio(baseline: bytes | None, current: bytes | None) -> float | None:
    """The fraction of pixels that differ, or ``None`` if it cannot be said.

    ``None`` means "no comparison", which is different from ``0.0`` meaning
    "identical" -- a UI that conflated them would report a missing baseline as
    a perfect match.
    """
    if not baseline or not current:
        return None
    try:
        from PIL import Image, ImageChops
    except ImportError:  # pragma: no cover - a deployment problem, not a run's
        log.debug("Pillow is not installed; screenshots will not be compared")
        return None

    try:
        with Image.open(io.BytesIO(baseline)) as first, Image.open(io.BytesIO(current)) as second:
            size = (COMPARE_WIDTH, COMPARE_HEIGHT)
            left = first.convert("RGB").resize(size)
            right = second.convert("RGB").resize(size)
            difference = ImageChops.difference(left, right)

            # Per pixel: the largest change across the three channels, so a
            # shift in one colour is not diluted by two that held still.
            changed = 0
            for pixel in difference.getdata():
                if max(pixel) > CHANNEL_TOLERANCE:
                    changed += 1
            return round(changed / (COMPARE_WIDTH * COMPARE_HEIGHT), 4)
    except Exception:  # noqa: BLE001 - a diff must never fail a run
        log.debug("could not compare screenshots", exc_info=True)
        return None


def describe(value: float | None) -> str:
    """The ratio in words, for a timeline that should not make people read
    four decimal places."""
    if value is None:
        return "no baseline"
    if value == 0:
        return "identical"
    if value < 0.01:
        return "almost identical"
    if value < 0.10:
        return "small differences"
    if value < 0.40:
        return "noticeably different"
    return "very different"


__all__ = ["CHANNEL_TOLERANCE", "COMPARE_HEIGHT", "COMPARE_WIDTH", "describe", "ratio"]
