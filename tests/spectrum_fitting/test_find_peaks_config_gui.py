"""The fitting-parameters dialog exposes the self-calibrating chain's three
switches (centre_method, dho_kernel, envelope_source) and greys out the
legacy camera-PSF constants whenever no selected path reads them."""
import os

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (  # noqa: E402
    find_peaks_reference_config, find_peaks_sample_config, sline_from_frame_config)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import (  # noqa: E402
    FindPeaksConfigDialog)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_switches_load_from_the_live_config(app):
    dlg = FindPeaksConfigDialog()
    assert dlg.sample_inputs["dho_kernel"].currentText() == find_peaks_sample_config.get().dho_kernel
    assert dlg.reference_inputs["centre_method"].currentText() == find_peaks_reference_config.get().centre_method
    assert dlg.global_inputs["envelope_source"].currentText() == sline_from_frame_config.get().envelope_source


def test_psf_constants_greyed_out_on_the_self_calibrating_chain(app):
    dlg = FindPeaksConfigDialog()
    dlg.reference_inputs["centre_method"].setCurrentText("template")
    dlg.sample_inputs["dho_kernel"].setCurrentText("measured")
    dlg.sample_inputs["fitting_model"].setCurrentText("dho_x_psf")
    assert not dlg.psf_constants_in_use()
    assert all(not dlg.global_inputs[k].isEnabled() for k in dlg.pr_field_names())
    # any legacy path switches them back on
    dlg.sample_inputs["dho_kernel"].setCurrentText("parametric")
    assert dlg.psf_constants_in_use()
    assert all(dlg.global_inputs[k].isEnabled() for k in dlg.pr_field_names())
    dlg.sample_inputs["dho_kernel"].setCurrentText("measured")
    dlg.reference_inputs["centre_method"].setCurrentText("parametric")
    assert dlg.psf_constants_in_use()
    dlg.reference_inputs["centre_method"].setCurrentText("template")
    dlg.sample_inputs["fitting_model"].setCurrentText("lorentzian_x_psf")
    assert dlg.psf_constants_in_use()


def test_apply_round_trips_the_switches(app, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    sample0 = find_peaks_sample_config.get()
    ref0 = find_peaks_reference_config.get()
    glob0 = sline_from_frame_config.get()
    dlg = FindPeaksConfigDialog()
    try:
        dlg.sample_inputs["dho_kernel"].setCurrentText("parametric")
        dlg.reference_inputs["centre_method"].setCurrentText("parametric")
        dlg.global_inputs["envelope_source"].setCurrentText("config")
        dlg.apply_config()
        assert find_peaks_sample_config.get().dho_kernel == "parametric"
        assert find_peaks_reference_config.get().centre_method == "parametric"
        assert sline_from_frame_config.get().envelope_source == "config"
    finally:
        find_peaks_sample_config.update(dho_kernel=sample0.dho_kernel)
        find_peaks_reference_config.update(centre_method=ref0.centre_method)
        sline_from_frame_config.update(envelope_source=glob0.envelope_source)
