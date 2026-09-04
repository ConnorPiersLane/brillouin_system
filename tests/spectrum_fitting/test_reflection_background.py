"""Tests for the ReflectionBackground template, its calibration-space mapper
and the 'reflection' background / prmr preset in SpectrumFitter.

The synthetic geometry mimics the real 4pk ROI: elastic lines at px ~60/~133,
the left(AS) order dispersing to higher px with offset g, the right(S) order
to lower px, different dispersions per order.
"""
import numpy as np
import pytest

from dataclasses import replace

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
)
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground,
    ReflectionBackgroundMapper,
    get_current_background,
    set_current_background,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    config_requires_reflection_background,
    normalize_model_name,
    resolved_background,
)

FREQS = np.linspace(4.0, 8.0, 41)

# Session A: the template's own geometry (x per order as a function of g).
XL_A = (60.0, 4.4, -0.05)   # x = c0 + c1*g + c2*g^2, increasing with g
XR_A = (133.0, -2.6, -0.01)  # decreasing with g

# Session B: after a "realignment" — shifted and stretched.
XL_B = (63.0, 4.62, -0.05)
XR_B = (135.5, -2.73, -0.01)


def x_of_g(track, g):
    c0, c1, c2 = track
    return c0 + c1 * np.asarray(g, float) + c2 * np.asarray(g, float) ** 2


def freq_polys(track_l, track_r, px_lo=0, px_hi=200):
    """px->g quadratics per order, the form a session calibration provides."""
    return (np.polyfit(x_of_g(track_l, FREQS), FREQS, 2),
            np.polyfit(x_of_g(track_r, FREQS), FREQS, 2))


def bump(px, c, w=1.2, a=1000.0):
    return a * np.exp(-0.5 * ((px - c) / w) ** 2)


def make_background(track_l=XL_A, track_r=XR_A, n_px=200, n_rows_frame=27):
    """A synthetic template: satellites at fixed offsets g on both orders."""
    px = np.arange(n_px, dtype=float)
    sline = np.zeros_like(px)
    for g, a in ((4.0, 1000.0), (6.8, 400.0)):
        sline += bump(px, float(x_of_g(track_l, g)), a=a)
        sline += bump(px, float(x_of_g(track_r, g)), a=a)
    # Spread the sline over rows with a Gaussian row profile centred off the
    # frame middle, so row selection actually matters.
    rows = np.arange(n_rows_frame, dtype=float)
    profile = np.exp(-0.5 * ((rows - 10.0) / 2.5) ** 2)
    profile /= profile.sum()
    frame = profile[:, None] * sline[None, :]
    return ReflectionBackground(
        frame=frame,
        cal_freqs=FREQS,
        cal_left_px=x_of_g(track_l, FREQS),
        cal_right_px=x_of_g(track_r, FREQS),
        meta={"synthetic": True},
    )


# ---------------- mapper: identity and realignment transfer ----------------

def test_identity_mapping_reproduces_own_sline():
    bg = make_background()
    mapper = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                        rows=None)
    px = bg.px
    R = mapper.render(px)
    sline = bg.sline(None)
    valid = R != 0.0
    assert valid.sum() > 20
    err = np.abs(R[valid] - sline[valid])
    assert err.max() < 0.02 * sline.max()


def test_realignment_transfer_lands_bumps_at_calibrated_pixels():
    # Template measured in session A, rendered onto session B: each satellite
    # must land where session B's calibration puts its frequency.
    bg = make_background()
    mapper = ReflectionBackgroundMapper(bg, freq_polys(XL_B, XR_B),
                                        rows=None)
    px = np.arange(0.0, 200.0)
    R = mapper.render(px)
    for track, g in ((XL_B, 4.0), (XL_B, 6.8), (XR_B, 4.0), (XR_B, 6.8)):
        expected = float(x_of_g(track, g))
        window = (px > expected - 4) & (px < expected + 4)
        w = R[window]
        centroid = float(np.sum(px[window] * w) / np.sum(w))
        assert abs(centroid - expected) <= 0.2, (g, expected, centroid)


def test_transfer_conserves_bump_integral():
    # The double Jacobian keeps counts-per-GHz: a satellite's integral in
    # counts survives the dispersion change to first order.
    bg = make_background()
    pxA = bg.px
    R_A = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                     rows=None).render(pxA)
    R_B = ReflectionBackgroundMapper(bg, freq_polys(XL_B, XR_B),
                                     rows=None).render(pxA)
    for track_a, track_b in ((XL_A, XL_B),):
        ca = float(x_of_g(track_a, 4.0))
        cb = float(x_of_g(track_b, 4.0))
        wa = (pxA > ca - 4) & (pxA < ca + 4)
        wb = (pxA > cb - 4) & (pxA < cb + 4)
        ia = float(np.sum(R_A[wa]))
        ib = float(np.sum(R_B[wb]))
        assert ia > 0 and abs(ib / ia - 1.0) < 0.06


def test_smaller_roi_needs_no_bookkeeping():
    # An 85-px axis whose calibration only covers the left order: the mapper
    # renders the in-range part and leaves the rest at zero.
    bg = make_background()
    fl, fr = freq_polys(XL_A, XR_A)
    px = np.arange(55.0, 140.0)
    R = mapper_render = ReflectionBackgroundMapper(
        bg, (fl, fr), rows=None).render(px)
    c40 = float(x_of_g(XL_A, 4.0))
    window = (px > c40 - 4) & (px < c40 + 4)
    assert R[window].max() > 0
    assert len(R) == len(px)


def test_sline_sums_exactly_the_requested_rows():
    # The template collapses over the SAME row indices the sample sline
    # sums (2026-08-26): the contamination in a sample sline is the part of
    # the pattern inside that band, wherever the pattern's own centre sits.
    bg = make_background()  # row profile centred on row 10
    full = bg.sline(None)
    on_band = bg.sline(range(5, 16))     # 11 rows around the pattern
    off_band = bg.sline(range(16, 27))   # 11 rows in the pattern's tail
    assert 0.9 < on_band.sum() / full.sum() <= 1.0
    assert off_band.sum() / full.sum() < 0.05
    assert np.allclose(bg.sline([9, 10, 11]),
                       bg.frame[9:12, :].sum(axis=0))
    # Rows beyond the template frame are ignored, not an error...
    assert np.allclose(bg.sline(range(20, 40)), bg.sline(range(20, 27)))
    # ...unless nothing overlaps at all.
    with pytest.raises(ValueError, match="exist in the template"):
        bg.sline(range(50, 60))


def test_mapper_rejects_calibration_without_polys():
    bg = make_background()
    with pytest.raises(ValueError, match="freq_left_peak"):
        ReflectionBackgroundMapper(bg, object())


def test_save_load_roundtrip(tmp_path):
    bg = make_background()
    path = tmp_path / "bg.npz"
    bg.save(path)
    back = ReflectionBackground.load(path)
    assert np.allclose(back.frame, bg.frame)
    assert np.allclose(back.cal_freqs, bg.cal_freqs)
    assert back.meta == bg.meta


def test_packaged_default_loads():
    bg = ReflectionBackground.load_default()
    assert bg.frame.ndim == 2
    assert len(bg.cal_freqs) >= 3
    # Identity render on its own calibration reproduces its own sline: feed
    # the mapper px->GHz polynomials fitted from the template's OWN
    # calibration points (the same form a session calibration provides).
    own_polys = (np.polyfit(bg.cal_left_px, bg.cal_freqs, 2),
                 np.polyfit(bg.cal_right_px, bg.cal_freqs, 2))
    rows = list(range(3, 14))
    mapper = ReflectionBackgroundMapper(bg, own_polys, rows=rows)
    R = mapper.render(bg.px)
    sline = bg.sline(rows)
    valid = R != 0.0
    assert valid.sum() > 10
    err = np.abs(R[valid] - sline[valid])
    assert err.max() < 0.01 * sline.max()


def test_current_background_registry():
    # Nothing loaded -> None: there is deliberately NO fallback to the
    # packaged default (a stale-alignment template applied silently is worse
    # than no reflection term).
    assert get_current_background() is None
    bg = make_background()
    set_current_background(bg)
    try:
        assert get_current_background() is bg
    finally:
        set_current_background(None)
    assert get_current_background() is None


# ---------------- fitter integration ----------------

SIGMA, TAU_L, TAU_R = 0.25, 0.4, 0.2


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
    )


def make_fitter(sample_model="prmr", reference_model="lorentzian_x_psf"):
    fitter = SpectrumFitter()
    fitter.update_sline_config(replace(
        fitter.sline_config, n_peaks=2,
        psf_sigma_left_px=SIGMA, psf_sigma_right_px=SIGMA,
        psf_tau_left_px=TAU_L, psf_tau_right_px=TAU_R))
    fitter.update_sample_config(make_config(sample_model))
    fitter.update_reference_config(make_config(reference_model))
    return fitter


def make_sample(R, s_true=0.05, seed=1):
    px = np.arange(0.0, 200.0)
    cen_l, cen_r = 78.0, 118.0
    truth = (
        psf_profile(px, 3000.0, cen_l, 1.0, SIGMA, TAU_L)
        + psf_profile(px, 3000.0, cen_r, 1.0, SIGMA, TAU_R)
        + np.where(px <= 98.0, 90.0, 60.0)
        + s_true * R
    )
    rng = np.random.default_rng(seed)
    return px, truth + rng.normal(0.0, 2.0, size=truth.shape), (cen_l, cen_r)


def test_prmr_preset_expands():
    cfg = make_config("prmr")
    assert cfg.fitting_model == "lorentzian_x_psf"
    assert cfg.background == "reflection"
    assert cfg.use_window is True
    assert cfg.beta == 3.0
    assert normalize_model_name("prmr") == ("lorentzian_x_psf", True)


def test_config_requires_reflection_background_helper():
    assert config_requires_reflection_background(make_config("prmr"))
    assert not config_requires_reflection_background(make_config("prm1"))
    # Direct assignment bypassing __post_init__ still resolves the preset.
    cfg = make_config("prm0")
    cfg.fitting_model = "prmr"
    assert resolved_background(cfg) == "reflection"
    assert config_requires_reflection_background(cfg)


def test_reflection_fit_recovers_truth_and_shared_scale():
    bg = make_background()
    R = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                   rows=None).render(np.arange(0.0, 200.0))
    px, sline, (cen_l, cen_r) = make_sample(R, s_true=0.05)
    fitter = make_fitter()
    result = fitter.fit(px, sline, is_reference_mode=False,
                        reflection_background=R)
    assert result.is_success
    assert "reflection" in result.model
    assert abs(result.left_peak_center_px - cen_l) < 0.05
    assert abs(result.right_peak_center_px - cen_r) < 0.05
    # Background parameters: [offset_left, offset_right, shared_s], appended
    # after the 6 peak parameters.
    o1, o2, s = result.parameters[6:9]
    assert abs(o1 - 90.0) < 5.0
    assert abs(o2 - 60.0) < 5.0
    assert abs(s - 0.05) < 0.01


def test_reflection_fit_without_template_warns_and_degrades():
    # No template passed -> warn and fit with per-peak flat offsets only
    # (no error, no silent default template).
    bg = make_background()
    R = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                   rows=None).render(np.arange(0.0, 200.0))
    px, sline, (cen_l, cen_r) = make_sample(R)
    fitter = make_fitter()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    # The recipe tag records the fit as it ran: windowed with the default
    # flat background (unnamed in the tag), NOT as a reflection fit.
    assert "reflection" not in result.model
    assert result.model == "2lorentzian_x_psf_window"
    # The unmodelled s*R structure is small; centres stay near truth.
    assert abs(result.left_peak_center_px - cen_l) < 0.5
    assert abs(result.right_peak_center_px - cen_r) < 0.5


def test_reflection_fit_rejects_mismatched_axis():
    bg = make_background()
    R = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                   rows=None).render(np.arange(0.0, 200.0))
    px, sline, _ = make_sample(R)
    fitter = make_fitter()
    with pytest.raises(ValueError, match="same pixel axis"):
        fitter.fit(px, sline, is_reference_mode=False,
                   reflection_background=R[:-5])


# ---------------- registration reach (g_margin_ghz) ----------------

def test_default_margin_clips_beyond_sweep():
    """A satellite beyond the calibrated sweep + default margin renders 0
    (registration not trusted out there), even though the template frame
    has real light at those pixels."""
    bg = make_background()
    # add a high-shift satellite at g = 9.6 on the left order
    x96 = float(x_of_g(XL_A, 9.6))
    bg = ReflectionBackground(
        frame=bg.frame + np.outer(
            np.exp(-0.5 * ((np.arange(bg.frame.shape[0]) - 10.0) / 2.5) ** 2),
            bump(bg.px, x96, a=500.0)),
        cal_freqs=bg.cal_freqs, cal_left_px=bg.cal_left_px,
        cal_right_px=bg.cal_right_px, meta=bg.meta)
    mapper = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                        rows=None)
    px = bg.px
    r = mapper.render(px)
    g = np.polyval(mapper.freq_left, px)
    beyond = (g > 8.7) & (px <= mapper._mid_px(px))
    assert np.all(r[beyond] == 0.0)


def test_wider_margin_renders_beyond_sweep():
    """g_margin_ghz = 2.0 reaches 10 GHz on a 4-8 sweep: the 9.6 GHz
    satellite is rendered at its calibrated pixel (identity mapping)."""
    base = make_background()
    x96 = float(x_of_g(XL_A, 9.6))
    bg = ReflectionBackground(
        frame=base.frame + np.outer(
            np.exp(-0.5 * ((np.arange(base.frame.shape[0]) - 10.0) / 2.5) ** 2),
            bump(base.px, x96, a=500.0)),
        cal_freqs=base.cal_freqs, cal_left_px=base.cal_left_px,
        cal_right_px=base.cal_right_px, meta=base.meta)
    mapper = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                        rows=None, g_margin_ghz=2.0)
    assert mapper.g_hi == pytest.approx(10.0)
    r = mapper.render(bg.px)
    i96 = int(round(x96))
    assert r[i96] > 100.0          # the satellite came through
    # and the default-margin mapper still clips it
    r0 = ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                    rows=None).render(bg.px)
    assert r0[i96] == 0.0


def test_margin_must_be_positive():
    bg = make_background()
    with pytest.raises(ValueError, match="positive"):
        ReflectionBackgroundMapper(bg, freq_polys(XL_A, XR_A),
                                   g_margin_ghz=0.0)


def test_config_carries_reflection_margin_default():
    from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
        FIND_PEAKS_TOML_PATH, load_config_section)
    cfg = load_config_section(FIND_PEAKS_TOML_PATH, "sample")
    assert cfg.reflection_margin_ghz == pytest.approx(0.7)
