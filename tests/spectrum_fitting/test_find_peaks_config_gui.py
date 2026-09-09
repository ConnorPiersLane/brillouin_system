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


def test_kernel_switch_loads_and_round_trips(app, monkeypatch):
    """kernel_source / kernel_file / kernel_file_lines (2026-09-09) ride in
    the [global] section like envelope_source; the file fields grey out on
    'scan'."""
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    glob0 = sline_from_frame_config.get()
    dlg = FindPeaksConfigDialog()
    assert dlg.global_inputs["kernel_source"].currentText() == glob0.kernel_source
    assert dlg.global_inputs["kernel_file_lines"].currentText() == glob0.kernel_file_lines
    assert dlg.global_inputs["kernel_file"].text() == str(glob0.kernel_file)
    try:
        dlg.global_inputs["kernel_source"].setCurrentText("scan")
        assert not dlg.global_inputs["kernel_file"].isEnabled()
        dlg.global_inputs["kernel_source"].setCurrentText("file")
        assert dlg.global_inputs["kernel_file"].isEnabled()
        dlg.global_inputs["kernel_file_lines"].setCurrentText("all")
        dlg.global_inputs["kernel_file"].setText("some/table.csv")
        assert dlg.global_inputs["kernel_match_shift_px_inner"].text() == str(glob0.kernel_match_shift_px_inner)
        dlg.global_inputs["kernel_match_shift_px_inner"].setText("0.02")
        dlg.apply_config()
        cfg = sline_from_frame_config.get()
        assert (cfg.kernel_source, cfg.kernel_file_lines, cfg.kernel_file) == ("file", "all", "some/table.csv")
        assert cfg.kernel_match_shift_px_inner == 0.02
    finally:
        sline_from_frame_config.update(kernel_source=glob0.kernel_source,
                                       kernel_file_lines=glob0.kernel_file_lines,
                                       kernel_file=glob0.kernel_file,
                                       kernel_match_shift_px_inner=glob0.kernel_match_shift_px_inner)


def test_load_and_compile_psf_buttons(app, monkeypatch, tmp_path):
    """'Load PSF...' takes a stored table and switches the source to file;
    'Compile from sweep...' builds one next to the .h5 and loads it."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    from test_template_nodes import _profiles
    from brillouin_system.spectrum_fitting.peak_fitting_config import find_peaks_config_gui as gui_mod
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    glob0 = sline_from_frame_config.get()
    table = tmp_path / "epsf_fine.csv"
    _profiles({0}).save(table, source="fine.h5")
    dlg = FindPeaksConfigDialog()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(table), ""))
    dlg._pick_kernel_file()
    assert dlg.global_inputs["kernel_file"].text() == str(table)
    assert dlg.global_inputs["kernel_source"].currentText() == "file"
    assert "left" in dlg._kernel_info.text() and "fine.h5" in dlg._kernel_info.text()
    # compile: the builder is stubbed (it needs a real sweep), the naming and
    # the load path are exercised
    h5 = tmp_path / "calibration_401.h5"
    h5.write_bytes(b"")
    calls = []

    def fake_save(cal, out, fitter=None, n_lines=None):
        calls.append((cal, str(out)))
        _profiles({0}).save(out, source=str(cal))
    import brillouin_system.spectrum_fitting.template_calibration as tc
    monkeypatch.setattr(tc, "save_epsf_file", fake_save)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(h5), ""))
    dlg._compile_kernel_file()
    assert calls == [(str(h5), str(tmp_path / "epsf_calibration_401.csv"))]
    assert dlg.global_inputs["kernel_file"].text() == str(tmp_path / "epsf_calibration_401.csv")
    assert (tmp_path / "epsf_calibration_401.csv").is_file()
    # nothing applied to the live config until Apply
    assert sline_from_frame_config.get().kernel_file == glob0.kernel_file


def test_analyzer_mismatch_dialog_routes_the_decision(app, monkeypatch):
    """The data analyzer's dialog answers the stored-PSF mismatch: Yes =
    recalculate from the scan's calibration, No = keep the stored PSF."""
    from PyQt5.QtWidgets import QMessageBox
    from types import SimpleNamespace
    import importlib
    fas = importlib.import_module("brillouin_system.analysis.fit_axial_scan")
    from brillouin_system.guis.data_analyzer.kernel_mismatch_dialog import (
        install_kernel_mismatch_dialog)
    bad = [SimpleNamespace(name="outer_left")]
    profiles = SimpleNamespace(path="table.csv")
    try:
        install_kernel_mismatch_dialog()
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        assert fas._kernel_mismatch_handler(bad, profiles) is True
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
        assert fas._kernel_mismatch_handler(bad, profiles) is False
    finally:
        fas.set_kernel_mismatch_handler(None)
