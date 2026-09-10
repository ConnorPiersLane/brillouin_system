"""File kernels (kernel_source = "file"): the DHO sample kernel of chosen
lines from a STORED ePSF node table, the axis, the other kernels and the
envelope from the scan's own calibration (spectrum_fitting/epsf.py
FileKernels / kernels_for_fit, sline config kernel_source / kernel_file /
kernel_file_lines).

Two synthetic node tables tell the paths apart: the scan's table carries a
WRONG profile (twice the instrument width), the stored one the TRUE
profile. A fit that reads the file recovers the acoustic width, one that
reads the scan does not.
"""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.dho import DhoAxes
from brillouin_system.spectrum_fitting.epsf import (
    Epsf, FileKernels, MatchLimits, OUTER_LINES, kernels_for_fit, load_epsf_file)
from brillouin_system.spectrum_fitting.measured_kernel import MeasuredKernel
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    SlineFromFrameConfig)

from synthetic_lines import CEN_LEFT, CEN_RIGHT, make_fitter
from test_template_nodes import (
    GRID, _fit, _profiles, _spectrum, _true_kernel, _wrong_kernel)


def _sline_config(**kw):
    return SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                                selected_rows=list(range(7, 20)), n_peaks=2, **kw)


def _tables(tmp_path):
    """(scan Epsf with the WRONG profile everywhere, path of a stored table
    with the TRUE profile at every node)."""
    scan = _profiles(set())
    scan._env = [0.01, -0.02]                      # the scan's own envelope
    true = _profiles({-3, -2, -1, 0, 1, 2, 3})
    true._env = [0.5, 0.5]                         # the fine sweep's, must never apply
    path = tmp_path / "fine.csv"
    true.save(path, source="calibration_401.h5")
    return scan, path


def test_config_validates_the_kernel_switch():
    cfg = _sline_config()
    assert (cfg.kernel_source, cfg.kernel_file, cfg.kernel_file_lines) == ("scan", "", "outer")
    with pytest.raises(ValueError, match="kernel_source"):
        _sline_config(kernel_source="table")
    with pytest.raises(ValueError, match="kernel_file_lines"):
        _sline_config(kernel_file_lines="inner")
    with pytest.raises(ValueError, match="needs kernel_file"):
        _sline_config(kernel_source="file")
    cfg = _sline_config(kernel_source="file", kernel_file="x.npz", kernel_file_lines="all")
    assert cfg.kernel_source == "file"


def test_file_kernels_take_the_named_lines_from_the_file_and_the_envelope_from_the_scan(tmp_path):
    scan, path = _tables(tmp_path)
    fine = load_epsf_file(path)
    assert fine.source == "calibration_401.h5"
    fk = FileKernels(scan, fine, ("right",), path=path)
    assert fk.names == scan.names and fk.n_lines == 2
    assert fk.file_lines == ("right",)
    assert (fk.source(0), fk.source(1)) == ("scan", "file")
    # the named line answers from the file (TRUE), the other from the scan (WRONG)
    assert np.allclose(fk.kernel_at(1, CEN_RIGHT).k, _true_kernel(1))
    assert np.allclose(fk.kernel_at(0, CEN_LEFT).k, _wrong_kernel(0))
    assert isinstance(fk.kernel(1, CEN_RIGHT + 0.4), MeasuredKernel)
    # the envelope is the scan's whatever the file stored
    assert fk.env_slope(0, CEN_LEFT) == pytest.approx(0.01)
    assert fk.env_slope(1, CEN_RIGHT) == pytest.approx(-0.02)
    assert fine.env_slope(1, CEN_RIGHT) == pytest.approx(0.5)
    assert fk.envelope is scan.envelope
    # axis-side attributes pass through to the scan
    assert fk.grid is scan.grid and fk.nodes is scan.nodes
    assert np.allclose(fk(0, CEN_LEFT, np.arange(70.0, 90.0)),
                       scan(0, CEN_LEFT, np.arange(70.0, 90.0)))
    with pytest.raises(ValueError, match="not fitted lines"):
        FileKernels(scan, fine, ("outer_left",))


def test_kernels_for_fit_follows_the_switch(tmp_path, caplog):
    scan, path = _tables(tmp_path)
    assert kernels_for_fit(scan, _sline_config()) is scan
    # "outer" on a two-line fit: nothing to take, the scan's kernels stay
    cfg = _sline_config(kernel_source="file", kernel_file=str(path), kernel_file_lines="outer")
    with caplog.at_level("WARNING"):
        assert kernels_for_fit(scan, cfg) is scan
    assert "using the scan's own kernels" in caplog.text
    cfg = _sline_config(kernel_source="file", kernel_file=str(path), kernel_file_lines="all")
    fk = kernels_for_fit(scan, cfg)
    assert isinstance(fk, FileKernels)
    assert fk.file_lines == ("left", "right")
    assert fk.path == str(path)
    # the stored table is read once per path
    assert kernels_for_fit(scan, cfg).file is fk.file
    with pytest.raises(FileNotFoundError, match="kernel_file"):
        kernels_for_fit(scan, replace(cfg, kernel_file=str(tmp_path / "missing.csv")))
    assert OUTER_LINES == ("outer_left", "outer_right")


def test_dho_fit_reads_the_file_kernels_per_frame(tmp_path):
    """The fitter's per-frame kernel lookup goes through the FileKernels:
    with the file (TRUE profile) the acoustic width is recovered, with the
    scan's own table (WRONG profile) it is not."""
    from brillouin_system.spectrum_fitting.measured_kernel import DX
    from synthetic_lines import G_INST, POLY_LEFT, POLY_RIGHT
    scan, path = _tables(tmp_path)
    fitter = make_fitter()
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT)
    snap = MeasuredKernel(u=GRID, k=_wrong_kernel(0), position_px=CEN_LEFT,
                          n_frames=14, g_median_px=2 * G_INST)
    frame, gam = _spectrum(0.7)
    cfg = replace(fitter.sline_config, kernel_source="file", kernel_file=str(path),
                  kernel_file_lines="all")
    fk = kernels_for_fit(scan, cfg)
    fit = _fit(fitter, frame, replace(base, kernel_left=snap, kernel_right=snap, profiles=fk))
    assert fit.is_success
    assert abs(fit.left_peak_width_px / gam[0] - 1.0) < 0.03
    assert abs(fit.right_peak_width_px / gam[1] - 1.0) < 0.03
    assert abs(fit.left_peak_center_px - (CEN_LEFT + 0.7)) < 0.03
    fit0 = _fit(fitter, frame, replace(base, kernel_left=snap, kernel_right=snap, profiles=scan))
    assert fit0.is_success
    assert fit0.left_peak_width_px / gam[0] < 0.9
    assert abs(fk.kernel_at(0, CEN_LEFT).k.sum() * DX - 1.0) < 1e-9


def test_dho_axes_carry_file_kernels_for_the_named_lines(tmp_path, monkeypatch):
    import importlib
    fas = importlib.import_module("brillouin_system.analysis.fit_axial_scan")
    from synthetic_lines import G_INST, POLY_LEFT, POLY_RIGHT
    scan, path = _tables(tmp_path)
    fitter = make_fitter()
    fitter.update_sline_config(replace(fitter.sline_config, kernel_source="file",
                                       kernel_file=str(path), kernel_file_lines="all"))
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT)
    calc = SimpleNamespace(p=SimpleNamespace(template_profiles=scan), dho_axes=lambda: base)
    monkeypatch.setattr("brillouin_system.spectrum_fitting.measured_kernel.sample_peak_positions",
                        lambda fitter, frame, n_peaks=2: (CEN_LEFT, CEN_RIGHT))
    # the WRONG scan profile fails the match check against the TRUE file on
    # purpose here; keep the file (handler says no fallback) to test the plumbing
    fas.set_kernel_mismatch_handler(lambda bad, prof: False)
    try:
        axes = fas._dho_axes_if_required(fitter, calc, SimpleNamespace(is_reference_mode=False),
                                         calibration_data=object(), first_frame=np.zeros((27, 200)))
    finally:
        fas.set_kernel_mismatch_handler(None)
    assert isinstance(axes.profiles, FileKernels)
    assert np.allclose(axes.kernel_left.k, _true_kernel(0))
    assert np.allclose(axes.kernel_right.k, _true_kernel(1))
    assert axes.env_slopes is None                 # scan.envelope is None here
    # switched off: the scan's own table, wrong profile and all
    fitter.update_sline_config(replace(fitter.sline_config, kernel_source="scan"))
    axes = fas._dho_axes_if_required(fitter, calc, SimpleNamespace(is_reference_mode=False),
                                     calibration_data=object(), first_frame=np.zeros((27, 200)))
    assert axes.profiles is scan
    assert np.allclose(axes.kernel_left.k, _wrong_kernel(0))


def test_match_limits_come_from_the_config_and_are_per_line(tmp_path):
    """The limits are fitting-config values (no constants in epsf.py); the
    centre-offset limit is tighter on the inner pair than on the outers."""
    cfg = _sline_config(kernel_match_shift_px_inner=0.01, kernel_match_shift_px_outer=0.03,
                        kernel_match_hwhm_fraction=0.06, kernel_match_rms_percent=4.0)
    lim = MatchLimits.from_config(cfg)
    assert (lim.shift_px("left"), lim.shift_px("outer_left")) == (0.01, 0.03)
    with pytest.raises(ValueError, match="kernel_match_rms_percent"):
        _sline_config(kernel_match_rms_percent=0.0)
    scan, path = _tables(tmp_path)
    fine = load_epsf_file(path)
    # a file whose profile is the scan's own shifted by +0.02 px
    shifted = _profiles(set())
    for line in range(2):
        shifted.profiles[line] = np.array([np.interp(GRID - 0.02, GRID, k, left=0.0, right=0.0)
                                           for k in shifted.profiles[line]])
    p2 = tmp_path / "shifted.csv"
    shifted.save(p2)
    fk = FileKernels(scan, load_epsf_file(p2), ("left", "right"))
    m = fk.check((CEN_LEFT, CEN_RIGHT), lim)
    assert all(abs(x.shift_px - 0.02) < 0.004 for x in m)
    assert not any(x.ok for x in m)                       # 0.02 > inner limit 0.01
    loose = MatchLimits(shift_px_inner=0.03, shift_px_outer=0.03, hwhm_fraction=0.06, rms_percent=4.0)
    assert all(x.ok for x in fk.check((CEN_LEFT, CEN_RIGHT), loose))
    assert "limit 0.010" in str(m[0])
    # the same table against itself passes and reports its limits
    same = FileKernels(fine, fine, ("left", "right"))
    assert all(x.ok for x in same.check((CEN_LEFT, CEN_RIGHT), lim))


def test_save_load_carries_provenance(tmp_path):
    e = _profiles({0})
    p = tmp_path / "t.csv"
    e.save(p, source="sweep.h5")
    e2 = Epsf.load(p)
    assert e2.source == "sweep.h5" and e2.path == str(p)
    assert np.allclose(e2.kernel(0, CEN_LEFT).k, _true_kernel(0))


def _axes_with(fitter, scan, path, monkeypatch, fas):
    from synthetic_lines import G_INST, POLY_LEFT, POLY_RIGHT
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT)
    calc = SimpleNamespace(p=SimpleNamespace(template_profiles=scan), dho_axes=lambda: base)
    monkeypatch.setattr("brillouin_system.spectrum_fitting.measured_kernel.sample_peak_positions",
                        lambda fitter, frame, n_peaks=2: (CEN_LEFT, CEN_RIGHT))
    return fas._dho_axes_if_required(fitter, calc, SimpleNamespace(is_reference_mode=False),
                                     calibration_data=object(), first_frame=np.zeros((27, 200)))


def test_mismatched_file_lines_fall_back_to_the_scan_kernel(tmp_path, monkeypatch):
    """The stored table is checked against the scan's own profile once per
    scan. A line that fails (here: the file's TRUE profile vs the scan's
    WRONG one, twice the width) is handed back to the scan's calibration
    kernel with a warning; a line that matches keeps the file kernel; an
    installed handler may veto the fallback (the analyzer's dialog)."""
    import importlib
    import logging
    fas = importlib.import_module("brillouin_system.analysis.fit_axial_scan")
    # scan: WRONG on both lines; file: TRUE on the left, WRONG on the right
    scan = _profiles(set())
    file = _profiles(set())
    file.profiles[0] = np.array([_true_kernel(0)] * len(file.nodes[0]))
    path = tmp_path / "fine.csv"
    file.save(path, source="fine.h5")
    fitter = make_fitter()
    fitter.update_sline_config(replace(fitter.sline_config, kernel_source="file",
                                       kernel_file=str(path), kernel_file_lines="all"))
    records = []
    h = logging.Handler(); h.emit = lambda r: records.append((r.levelno, r.getMessage()))
    fas.log.addHandler(h)
    level0 = fas.log.level
    fas.log.setLevel(logging.INFO)
    try:
        axes = _axes_with(fitter, scan, path, monkeypatch, fas)
        # left mismatched -> scan kernel (WRONG); right matched -> file kernel (= WRONG too, but from the file)
        assert isinstance(axes.profiles, FileKernels)
        assert axes.profiles.file_lines == ("right",)
        assert np.allclose(axes.kernel_left.k, _wrong_kernel(0))
        assert any(lvl == logging.WARNING and "MISMATCH" in m and "left" in m for lvl, m in records)
        assert any("recalculated from this scan" in m for _, m in records)
        # a handler that says "keep the file" wins
        records.clear()
        seen = []
        fas.set_kernel_mismatch_handler(lambda bad, prof: (seen.append([m.name for m in bad]), False)[1])
        try:
            axes = _axes_with(fitter, scan, path, monkeypatch, fas)
        finally:
            fas.set_kernel_mismatch_handler(None)
        assert seen == [["left"]]
        assert axes.profiles.file_lines == ("left", "right")
        assert np.allclose(axes.kernel_left.k, _true_kernel(0))
        assert any("kept on user decision" in m for _, m in records)
        # every file line failing -> the scan's own Epsf, plain
        scan2 = _profiles(set())
        file2 = _profiles({-3, -2, -1, 0, 1, 2, 3})
        path2 = tmp_path / "fine2.csv"
        file2.save(path2)
        fitter.update_sline_config(replace(fitter.sline_config, kernel_file=str(path2)))
        axes = _axes_with(fitter, scan2, path2, monkeypatch, fas)
        assert axes.profiles is scan2
        # matching lines are logged at INFO, no warning
        records.clear()
        fitter.update_sline_config(replace(fitter.sline_config, kernel_file=str(path)))
        _axes_with(fitter, file, path, monkeypatch, fas)      # the file against itself
        assert any(lvl == logging.INFO and "[kernels]" in m for lvl, m in records)
        assert not any(lvl == logging.WARNING for lvl, _ in records)
    finally:
        fas.log.removeHandler(h)
        fas.log.setLevel(level0)
