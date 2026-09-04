"""
Automatic selection of the camera rows that are summed into the spectral line.

The spectrum runs along x; the rows (y) hold the same spectrum repeated with a
small TILT, because a grating disperses in the orthogonal direction before the
VIPA. Summing a band of rows therefore mixes slightly different peak
positions, and WHICH rows are summed shifts the fitted peaks:

    measured 2026-07 (2026-7-9 session, 13-row band): moving the band by one
    row shifts the individual peaks by ~3-4 MHz, in OPPOSITE frequency
    directions (both move the same way in pixels, so the peak DISTANCE is
    nearly immune: < 1 MHz for +-2 rows).

Two consequences drive the design here:

1. The band must be picked from a STABLE statistic. Choosing "the N contiguous
   rows with the most signal" (argmax of a sliding sum) flips between two
   nearly-equal windows on noise: on real data it picked start row 4 in 47/50
   sample frames but start row 5 in 25/41 calibration frames of the same scan.
   The intensity-weighted CENTROID is smooth and agreed to 0.03 rows between
   calibration and sample light (10.11 vs 10.08), so it rounds to the same
   band every time. That is what is used below.

2. The same band must be used for a scan's calibration and its sample frames.
   A one-row mismatch puts ~3-4 MHz of opposite-sign bias into the two peaks,
   i.e. ~7 MHz into the left-right comparison.
"""
import numpy as np


def row_profile(frames):
    """Background-subtracted signal per row, averaged over frames.

    The per-row median across columns is used as that row's background, so a
    sloped or structured baseline does not bias the profile.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 2:
        frames = frames[None, :, :]
    prof = np.zeros(frames.shape[1], dtype=float)
    for fr in frames:
        prof += (fr - np.median(fr, axis=1, keepdims=True)).sum(axis=1)
    return prof / len(frames)


def row_centroid(frames, half_window: int | None = None) -> float:
    """Intensity-weighted centroid row of the spectral line.

    The centroid is taken over a window around the brightest row so that far
    wings and noise cannot pull it; positive weights only.
    """
    prof = row_profile(frames)
    weights = np.clip(prof, 0.0, None)
    if weights.sum() <= 0:
        raise ValueError("Cannot locate the spectral line: no positive signal "
                         "in the row profile.")
    peak_row = int(np.argmax(weights))
    if half_window is None:
        half_window = max(int(round(len(prof) / 4)), 3)
    lo = max(peak_row - half_window, 0)
    hi = min(peak_row + half_window + 1, len(prof))
    rows = np.arange(lo, hi, dtype=float)
    w = weights[lo:hi]
    return float((rows * w).sum() / w.sum())


def select_rows(frames, n_rows: int) -> list[int]:
    """Return the n_rows contiguous rows centred on the line.

    frames may be a single frame or a stack; a stack is preferable because the
    centroid is then measured at higher SNR. Pass the SAME result to the
    calibration and the sample frames of a scan.
    """
    frames = np.asarray(frames, dtype=float)
    height = frames.shape[-2]
    n_rows = int(n_rows)
    if not 1 <= n_rows <= height:
        raise ValueError(
            f"n_rows must be between 1 and the frame height ({height}), "
            f"got {n_rows}."
        )
    centroid = row_centroid(frames)
    start = int(round(centroid - (n_rows - 1) / 2.0))
    start = int(np.clip(start, 0, height - n_rows))
    return list(range(start, start + n_rows))


def captured_fraction(frames, rows) -> float:
    """Fraction of the total row-profile signal inside the given rows."""
    prof = np.clip(row_profile(frames), 0.0, None)
    total = prof.sum()
    if total <= 0:
        return 0.0
    return float(prof[list(rows)].sum() / total)
