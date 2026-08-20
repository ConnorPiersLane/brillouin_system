"""Tests for brillouin_peak_fitting.calibration.background_reflection.

Run with paper_repo/src on PYTHONPATH.
"""
import numpy as np

from brillouin_peak_fitting.calibration import (
    ReflectionBackground,
    ReflectionBackgroundMapper,
)


def make_template():
    """Synthetic template with exactly LINEAR tracks, so the quadratic
    g<->px fits are exact and the identity mapping has no fit error."""
    px = np.arange(300, dtype=float)
    # Left(AS) order: g = (200 - px) / 10; right(S): g = (px - 100) / 10.
    # The tracks cross at px 150 (g = 5), inside the axis.
    freqs = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    left_px = 200.0 - 10.0 * freqs
    right_px = 100.0 + 10.0 * freqs

    sline = (5.0
             + 200.0 * np.exp(-0.5 * ((px - 130.0) / 3.0) ** 2)
             + 120.0 * np.exp(-0.5 * ((px - 170.0) / 3.0) ** 2))
    frame = np.tile(sline / 5.0, (5, 1))
    return ReflectionBackground(frame=frame, cal_freqs=freqs,
                                cal_left_px=left_px, cal_right_px=right_px)


def test_identity_mapping_reproduces_the_template():
    """Rendered onto its own session (same calibration, same axis), the
    template must come back unchanged wherever the tracks are valid."""
    bg = make_template()
    mapper = ReflectionBackgroundMapper(bg, bg.own_freq_polys())
    out = mapper.render(bg.px)
    sline = bg.sline()

    # Valid region: g within [g_lo, g_hi] on each side of the split (px 150).
    px = bg.px
    g_left = (200.0 - px) / 10.0
    g_right = (px - 100.0) / 10.0
    valid = np.where(px <= 150.0,
                     (g_left >= bg.g_lo) & (g_left <= bg.g_hi),
                     (g_right >= bg.g_lo) & (g_right <= bg.g_hi))
    # [3.3, 8.7] GHz on each 10 px/GHz track: ~38 + ~37 px, both bumps inside.
    assert valid.sum() > 70
    np.testing.assert_allclose(out[valid], sline[valid], rtol=1e-9)
    # Outside the swept range (+ margin) nothing is invented.
    assert np.all(out[~valid] == 0.0)


def test_roi_crop_needs_no_bookkeeping():
    """A cropped axis is just a slice of the full render."""
    bg = make_template()
    mapper = ReflectionBackgroundMapper(bg, bg.own_freq_polys())
    full = mapper.render(bg.px)
    crop = np.arange(120, 190, dtype=float)
    np.testing.assert_allclose(mapper.render(crop), full[120:190], rtol=1e-9)


def test_dispersion_change_conserves_counts():
    """A target session with different dispersion must conserve the counts
    under each order (density mapping, not value copying)."""
    bg = make_template()
    # Target session: 20 px/GHz instead of 10, tracks crossing at px 300.
    freq_left = np.polyfit([400.0 - 20.0 * g for g in (4.0, 6.0, 8.0)],
                           [4.0, 6.0, 8.0], 1)
    freq_right = np.polyfit([200.0 + 20.0 * g for g in (4.0, 6.0, 8.0)],
                            [4.0, 6.0, 8.0], 1)
    mapper = ReflectionBackgroundMapper(bg, (freq_left, freq_right))
    px_target = np.arange(600, dtype=float)
    out = mapper.render(px_target)

    # The left-order bump (template px 130 -> g 7.0 -> target px 260).
    assert abs(px_target[np.argmax(out[:300])] - 260.0) < 1.0
    # Counts conserved: 2x the pixels per GHz at half the value each, so the
    # sum over the same g window matches the template (density mapping, not
    # value copying — a value copy would double the counts here).
    sline = bg.sline()
    tmpl_sum = sline[(bg.px >= 110) & (bg.px <= 150)].sum()      # g 5..9 left
    trgt_sum = out[(px_target >= 220) & (px_target <= 300)].sum()
    assert abs(trgt_sum / tmpl_sum - 1.0) < 0.02


def test_save_load_roundtrip(tmp_path):
    bg = make_template()
    bg.meta = {"source": "synthetic"}
    path = tmp_path / "bg.npz"
    bg.save(path)
    back = ReflectionBackground.load(path)
    np.testing.assert_array_equal(back.frame, bg.frame)
    np.testing.assert_array_equal(back.cal_freqs, bg.cal_freqs)
    assert back.meta == {"source": "synthetic"}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_identity_mapping_reproduces_the_template()
    test_roi_crop_needs_no_bookkeeping()
    test_dispersion_change_conserves_counts()
    with tempfile.TemporaryDirectory() as d:
        test_save_load_roundtrip(Path(d))
    print("ALL BACKGROUND-REFLECTION TESTS PASSED")
