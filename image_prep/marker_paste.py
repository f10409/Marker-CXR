"""Paste marker stencils from the bank into a radiograph.

Each paste gets the stencil's physical size (mm), plus random scale,
rotation, brightness and noise. Text stencils are pinned at the film's
ceiling value, as burned-in text is; physical markers get a brightness from
their bank ``peak_frac`` mapped into the target film's range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

import marker_bank_io as mb

_BBOX_ALPHA_CUT = 0.05  # separates the visible mark from interpolation dust at a zoomed/rotated edge


def _is_text(p: mb.Pattern) -> bool:
    """Text vs. physical marker: the bank label if set, else classify()."""
    return p.label.startswith(mb.TEXT) if p.label else mb.classify(p) == mb.TEXT


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _tight_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    """Pixel bbox of alpha's visible part (zoom/rotate add transparent border),
    or None if nothing clears the cut."""
    ys, xs = np.nonzero(alpha > _BBOX_ALPHA_CUT)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def paste_random(
    path: str | Path | mb.Donor,
    real_bboxes: Sequence[tuple[float, float, float, float]],
    pats: Sequence[mb.Pattern],
    *,
    n_range: tuple[int, int] = (1, 3),
    scale_jitter: tuple[float, float] = (0.9, 1.1),
    rotate_deg: tuple[float, float] = (-15.0, 15.0),
    intensity_jitter: float = 0.1,
    noise_std: float = 0.02,
    rng: np.random.Generator | int | None = None,
    max_tries: int = 20,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Paste a random number of bank stencils onto the film at ``path``.

    Args:
        path: target DICOM, or an opened ``Donor`` of it (read at full resolution).
        real_bboxes: markers already on the film, (x0, y0, x1, y1) normalized;
            pastes avoid them. ``()`` for a marker-free film.
        pats: stencils, e.g. ``load_bank(bank_dir)``.
        n_range: inclusive (min, max) number of stencils to paste.
        scale_jitter: random multiplier on the mm-correct size.
        rotate_deg: random rotation range in degrees.
        intensity_jitter: relative jitter on a physical marker's brightness
            (not applied to text, which is pinned at the film's ceiling).
        noise_std: Gaussian noise on physical markers, as a fraction of the
            film's range (not applied to text).
        rng: seed or numpy Generator. The same seed gives the same result.
        max_tries: placement attempts per stencil to avoid overlap.

    Returns:
        (pasted float image at full resolution, one (x0, y0, x1, y1) box per
        pasted stencil, normalized and rounded to 3 decimals). Stencils too big
        for the film are skipped, so there can be fewer boxes than drawn.
    """
    donor = path if isinstance(path, mb.Donor) else mb.Donor(path)
    image, _ = donor.screen_array(max_side=None)
    mm_per_px = donor.mm_per_px

    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    out = np.array(image, dtype=np.float32, copy=True)
    H, W = out.shape

    lo = float(np.percentile(out, 1.0))
    hi = float(out.max())
    span = max(hi - lo, 1e-6)
    img_min = float(out.min())  # true floor -- noise must not paste a pixel darker than any real one

    n = int(rng.integers(n_range[0], n_range[1] + 1))
    picks = rng.permutation(len(pats))[:n]

    placed: list[tuple[int, int, int, int]] = [
        (int(rx0 * W), int(ry0 * H), int(rx1 * W), int(ry1 * H))
        for rx0, ry0, rx1, ry1 in real_bboxes
    ]
    boxes: list[tuple[float, float, float, float]] = []
    for idx in picks:
        p = pats[idx]
        is_text = _is_text(p)

        phys_scale = (
            p.mm_per_px / mm_per_px
            if np.isfinite(p.mm_per_px) and np.isfinite(mm_per_px) and mm_per_px > 0
            else 1.0
        )
        scale = phys_scale * rng.uniform(*scale_jitter)
        alpha = np.clip(ndimage.zoom(p.alpha, scale, order=1), 0.0, 1.0)

        angle = rng.uniform(*rotate_deg)
        if angle:
            alpha = np.clip(
                ndimage.rotate(alpha, angle, reshape=True, order=1, mode="constant", cval=0.0),
                0.0, 1.0,
            )

        tight = _tight_bbox(alpha)
        h, w = alpha.shape
        if tight is None or h >= H or w >= W:
            continue

        if is_text:
            v = hi  # text sits at the film's ceiling
        else:
            frac = float(np.clip(
                p.peak_frac * rng.uniform(1 - intensity_jitter, 1 + intensity_jitter), 0.0, 1.0,
            ))
            v = lo + frac * span

        tx0, ty0, tx1, ty1 = tight
        rect = None
        for _ in range(max_tries):
            y0 = int(rng.integers(0, H - h + 1))
            x0 = int(rng.integers(0, W - w + 1))
            rect = (x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1)
            if not any(_overlaps(rect, r) for r in placed):
                break
        placed.append(rect)

        composite = out[y0 : y0 + h, x0 : x0 + w] * (1 - alpha) + v * alpha
        if not is_text and noise_std > 0:
            noisy = composite + rng.normal(0.0, noise_std * span, size=alpha.shape) * alpha
            # Gaussian noise is unbounded; a real film's pixels never exceed its own
            # min/max, so an un-clamped draw could paste a value no real image has.
            composite = np.clip(noisy, img_min, hi)
        out[y0 : y0 + h, x0 : x0 + w] = composite

        x0b, y0b, x1b, y1b = rect
        boxes.append((round(x0b / W, 3), round(y0b / H, 3), round(x1b / W, 3), round(y1b / H, 3)))

    return out, boxes
