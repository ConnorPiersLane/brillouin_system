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
    Epsf, FileKernels, OUTER_LINES, kernels_for_fit, load_epsf_file)
from brillouin_system.spectrum_fitting.measured_kernel import MeasuredKernel
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    SlineFromFrameConfig)

from test_measured_kernel import CEN_LEFT, CEN_RIGHT, make_fitter
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
    from brillouin_system.spectrum_fitting.psf import DX
    from test_measured_kernel import G_INST, POLY_LEFT, POLY_RIGHT
    scan, path = _tables(tmp_path)
    fitter = make_fitter()
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT,
                   instrument_width_left_poly=np.array([G_INST]),
                   instrument_width_right_poly=np.array([G_INST]))
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


def test_dho_axes_refuse_file_kernels_on_a_parametric_axis(tmp_path):
    from brillouin_system.analysis.fit_axial_scan import _dho_axes_if_required
    from test_measured_kernel import G_INST, POLY_LEFT, POLY_RIGHT
    _, path = _tables(tmp_path)
    fitter = make_fitter()
    fitter.update_sample_config(replace(fitter.sample_config, dho_kernel="measured"))
    fitter.update_sline_config(replace(fitter.sline_config, kernel_source="file",
                                       kernel_file=str(path), kernel_file_lines="all"))
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT,
                   instrument_width_left_poly=np.array([G_INST]),
                   instrument_width_right_poly=np.array([G_INST]))
    calc = SimpleNamespace(p=SimpleNamespace(), dho_axes=lambda: base)   # no template_profiles
    with pytest.raises(ValueError, match="centre_method = 'template'"):
        _dho_axes_if_required(fitter, calc, SimpleNamespace(is_reference_mode=False),
                              calibration_data=object(), first_frame=np.zeros((27, 200)))


def test_dho_axes_carry_file_kernels_for_the_named_lines(tmp_path, monkeypatch):
    import importlib
    fas = importlib.import_module("brillouin_system.analysis.fit_axial_scan")
    from test_measured_kernel import G_INST, POLY_LEFT, POLY_RIGHT
    scan, path = _tables(tmp_path)
    fitter = make_fitter()
    fitter.update_sample_config(replace(fitter.sample_config, dho_kernel="measured"))
    fitter.update_sline_config(replace(fitter.sline_config, kernel_source="file",
                                       kernel_file=str(path), kernel_file_lines="all"))
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT,
                   instrument_width_left_poly=np.array([G_INST]),
                   instrument_width_right_poly=np.array([G_INST]))
    calc = SimpleNamespace(p=SimpleNamespace(template_profiles=scan), dho_axes=lambda: base)
    monkeypatch.setattr("brillouin_system.spectrum_fitting.measured_kernel.sample_peak_positions",
                        lambda fitter, frame, n_peaks=2: (CEN_LEFT, CEN_RIGHT))
    axes = fas._dho_axes_if_required(fitter, calc, SimpleNamespace(is_reference_mode=False),
                                     calibration_data=object(), first_frame=np.zeros((27, 200)))
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


def test_save_load_carries_provenance(tmp_path):
    e = _profiles({0})
    p = tmp_path / "t.csv"
    e.save(p, source="sweep.h5")
    e2 = Epsf.load(p)
    assert e2.source == "sweep.h5" and e2.path == str(p)
    assert np.allclose(e2.kernel(0, CEN_LEFT).k, _true_kernel(0))
