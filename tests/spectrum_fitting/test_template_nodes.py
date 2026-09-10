"""The sample fit takes its kernel from the template NODE table, per frame.

Epsf.kernel_at must blend the two 1-px node profiles bracketing
the position (no restack, continuous along the track), and a DHO fit given
the node table must pick, for every frame, the kernel at that frame's found
peak position: a scan whose peaks move (cornea depth) is fitted with the
profile measured where each peak is.
"""
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.dho import DhoAxes, dho_profile
from brillouin_system.spectrum_fitting.measured_kernel import MeasuredKernel
from brillouin_system.spectrum_fitting.measured_kernel import DX
from brillouin_system.spectrum_fitting.epsf import Epsf

from synthetic_lines import (
    CEN_LEFT, CEN_RIGHT, FLOOR, G_INST, GRID, POLY_LEFT, POLY_RIGHT, PX, SIGMA,
    SLOPE_LEFT, SLOPE_RIGHT, TAU, frame_from_sline, make_fitter, measured_kernel,
    unit_kernel)


def _true_kernel(idx):
    return unit_kernel(G_INST, SIGMA[idx], TAU[idx])


def _wrong_kernel(idx):
    # twice the instrument width: a fit with this kernel returns a visibly
    # wrong acoustic width, so a test can tell which node was used
    return unit_kernel(2.0 * G_INST, SIGMA[idx], TAU[idx])


def _profiles(good_offsets):
    """Two-line node table, nodes every 1 px over +-3 px around each line;
    the TRUE kernel at the node offsets in `good_offsets`, the WRONG one
    everywhere else."""
    nodes, profiles, counts = [], [], []
    for idx, c0 in ((0, CEN_LEFT), (1, CEN_RIGHT)):
        nd = np.array([c0 + d for d in range(-3, 4)], dtype=float)
        prof = {float(n): (_true_kernel(idx) if int(round(n - c0)) in good_offsets
                           else _wrong_kernel(idx)) for n in nd}
        nodes.append(nd)
        profiles.append(prof)
        counts.append({float(n): 14 for n in nd})
    return Epsf.from_table(GRID, nodes, profiles, counts, env_slope=[0.0, 0.0])


def test_kernel_at_blends_the_two_bracketing_nodes():
    tp = _profiles({0})
    k = tp.kernel_at(0, CEN_LEFT)                 # exactly on a node
    assert isinstance(k, MeasuredKernel)
    assert k.position_px == CEN_LEFT
    assert k.n_frames == 14
    assert np.allclose(k.k, _true_kernel(0))
    k4 = tp.kernel_at(0, CEN_LEFT + 0.4)          # 60 % node 0, 40 % node +1
    assert np.allclose(k4.k, 0.6 * _true_kernel(0) + 0.4 * _wrong_kernel(0))
    assert abs(k4.k.sum() * DX - 1.0) < 1e-9      # still unit area
    assert np.allclose(tp.kernel_at(0, CEN_LEFT + 1.0).k, _wrong_kernel(0))
    # beyond the ladder: the end node alone, until coverage runs out
    assert np.allclose(tp.kernel_at(0, CEN_LEFT + 3 + 2.0).k, _wrong_kernel(0))
    with pytest.raises(ValueError, match="No template node"):
        tp.kernel_at(0, CEN_LEFT + 3 + 2.6)


def _spectrum(dc):
    """Noise-free DHO pair with both peaks displaced by dc px."""
    gamma_ghz = 0.143
    gam_px = (gamma_ghz / SLOPE_LEFT, gamma_ghz / abs(SLOPE_RIGHT))
    cl, cr = CEN_LEFT + dc, CEN_RIGHT + dc
    reach = lambda c: np.abs(PX - c) <= 15.0
    sline = (FLOOR
             + dho_profile(PX, 3400.0, cl, gam_px[0], POLY_LEFT,
                           measured_kernel(G_INST, SIGMA[0], TAU[0])) * reach(cl)
             + dho_profile(PX, 3400.0, cr, gam_px[1], POLY_RIGHT,
                           measured_kernel(G_INST, SIGMA[1], TAU[1])) * reach(cr))
    return frame_from_sline(sline), gam_px


def _fit(fitter, frame, axes):
    px, s = fitter.get_px_sline_from_image(frame)
    return fitter.fit(np.asarray(px, float), np.asarray(s, float),
                      is_reference_mode=False, dho_axes=axes)


def test_fit_picks_the_node_at_each_frame_peak_position():
    fitter = make_fitter()
    base = DhoAxes(freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT)
    # the TRUE profile sits at nodes +1..+3 px only (the blend around +2
    # then stays true whatever the finder's sub-pixel error); the
    # first-frame snapshot kernels (kernel_left/right) are the WRONG ones
    tp = _profiles({1, 2, 3})
    wrong = MeasuredKernel(u=GRID, k=_wrong_kernel(0), position_px=CEN_LEFT, n_frames=14, g_median_px=2 * G_INST)
    axes = replace(base, kernel_left=wrong, kernel_right=wrong, profiles=tp)

    # frame A: peaks at +2 px -> node +2 -> true kernel -> width recovered
    frame, gam = _spectrum(2.0)
    fit = _fit(fitter, frame, axes)
    assert fit.is_success
    assert abs(fit.left_peak_width_px / gam[0] - 1.0) < 0.03
    assert abs(fit.right_peak_width_px / gam[1] - 1.0) < 0.03
    assert abs(fit.left_peak_center_px - (CEN_LEFT + 2.0)) < 0.03

    # frame B, same scan: peaks at 0 px -> node 0 -> wrong kernel -> the
    # acoustic width absorbs the missing instrument width (well off)
    frame0, gam0 = _spectrum(0.0)
    fit0 = _fit(fitter, frame0, axes)
    assert fit0.is_success
    assert fit0.left_peak_width_px / gam0[0] < 0.9

    # without the node table the snapshot kernel is used for every frame
    fixed = replace(axes, profiles=None)
    fit_fixed = _fit(fitter, frame, fixed)
    assert fit_fixed.left_peak_width_px / gam[0] < 0.9
