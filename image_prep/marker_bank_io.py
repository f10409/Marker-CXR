"""Load the marker bank and read DICOMs for pasting.

The bank (on PhysioNet with Marker-CXR) is a folder of marker stencils cut
from MIMIC-CXR films:

    bank/
      index.jsonl       one JSON line per stencil (metadata below)
      patterns/*.npy    alpha mask per stencil, 0-1 coverage per pixel

A stencil is pasted with ``out = bg * (1 - alpha) + v * alpha``, where ``bg``
is the target film and ``v`` a brightness taken from the target's own range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT = "text"
MARKER = "marker"

# peak_frac >= this means the stencil's core was the brightest value in its film.
_AT_CEILING = 0.9999


@dataclass
class Pattern:
    """One marker stencil.

    alpha: coverage per pixel, 0-1.
    mm_per_px: physical scale of ``alpha`` (from ImagerPixelSpacing), so a
        15 mm marker stays 15 mm in a film with a different pixel spacing.
    height_mm, width_mm: stencil size in mm.
    peak_frac: marker core brightness as a fraction of its donor film's range.
    bg_frac: background under the marker, same 0-1 scale.
    clipped: fraction of stencil pixels at the donor's ceiling.
    fill: stencil pixels above 0.5 / box pixels.
    kind, source, score: detector metadata, not used for pasting.
    path: donor DICOM, relative to MIMIC-CXR files/.
    bbox: marker box in the donor, normalized (x0, y0, x1, y1).
    label: "text" (burned-in annotation) or "marker" (physical lead marker).
    """

    alpha: np.ndarray
    mm_per_px: float
    height_mm: float
    width_mm: float
    peak_frac: float
    bg_frac: float
    clipped: float
    fill: float
    kind: str
    source: str
    score: float
    path: str
    bbox: tuple[float, float, float, float]
    label: str = ""


def classify(p: Pattern | dict) -> str:
    """"text" if the stencil's core reached its film's maximum value, else
    "marker". Burned-in text is written at the pixel ceiling; lead is not."""
    peak = p.peak_frac if isinstance(p, Pattern) else p["peak_frac"]
    return TEXT if peak >= _AT_CEILING else MARKER


def load_bank(bank_dir: str | Path) -> list[Pattern]:
    """Read every stencil listed in index.jsonl whose .npy file exists."""
    bank_dir = Path(bank_dir)
    pats = []
    for line in (bank_dir / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        f = bank_dir / rec.pop("file")
        if not f.exists():
            continue
        rec.pop("shape", None)
        rec["bbox"] = tuple(rec["bbox"])
        pats.append(Pattern(alpha=np.load(f), **rec))
    return pats


class Donor:
    """One DICOM, decoded once: pixel array, LUTs and pixel spacing."""

    __slots__ = ("d", "px", "slope", "inter", "mono1")

    def __init__(self, path: str | Path):
        import pydicom

        self.d = pydicom.dcmread(str(path))
        px = self.d.pixel_array
        if px.ndim == 3:  # multiframe: first frame
            px = px[0]
        self.px = px
        self.slope = float(getattr(self.d, "RescaleSlope", 1) or 1)
        self.inter = float(getattr(self.d, "RescaleIntercept", 0) or 0)
        self.mono1 = (
            getattr(self.d, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1"
        )

    def _lut(self, a: np.ndarray) -> np.ndarray:
        """Modality LUT then VOI LUT."""
        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:  # pydicom < 3
            from pydicom.pixel_data_handlers.util import apply_voi_lut

        if self.slope != 1.0 or self.inter != 0.0:
            a = a * self.slope + self.inter
        try:
            return np.asarray(apply_voi_lut(a, self.d), dtype=np.float32)
        except Exception:
            return np.asarray(a, dtype=np.float32)  # malformed window: use raw values

    @property
    def mm_per_px(self) -> float:
        ps = getattr(self.d, "ImagerPixelSpacing", None) or getattr(
            self.d, "PixelSpacing", None
        )
        return float(ps[0]) if ps else float("nan")

    def screen_array(self, max_side: int | None = 768) -> tuple[np.ndarray, tuple[int, int]]:
        """Windowed, bone-bright float array, optionally decimated so the short
        side is about ``max_side``. ``max_side=None`` = full resolution.
        Returns (array, (W, H) of the original)."""
        H, W = self.px.shape
        step = max(1, int(min(self.px.shape) // max_side)) if max_side else 1
        a = self._lut(np.ascontiguousarray(self.px[::step, ::step]))
        if self.mono1:
            a = float(a.max()) - a
        return a, (W, H)
