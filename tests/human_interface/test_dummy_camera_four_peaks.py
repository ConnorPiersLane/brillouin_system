"""The DummyCamera (the GUI's zero-hardware dev mode) must serve frames
that work under the four-peak standard AND under n_peaks=2: every frame
carries all four VIPA orders, the outer pair at ~60% amplitude, so the
amplitude ranking keeps the inner pair on a two-peak fit."""
from dataclasses import replace

from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


class _FakeMicrowave:
    freq = 5.75

    def get_frequency(self):
        return self.freq


def _camera() -> DummyCamera:
    cam = DummyCamera(microwave=_FakeMicrowave())
    cam.set_verbose(False)
    cam.set_exposure_time(0.05)
    return cam


def _fitter(n_peaks: int) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sline_config(replace(fitter.sline_config, n_peaks=n_peaks))
    # flat sample background: this test pins the dummy's peak layout, not
    # the reflection-template plumbing; reference threshold scales with the
    # shortened exposure (the live 800 assumes 0.3 s)
    fitter.update_sample_config(replace(
        fitter.sample_config, background="flat"))
    fitter.update_reference_config(replace(
        fitter.reference_config, min_peak_height=100))
    return fitter


def test_dummy_frames_fit_four_peak():
    cam = _camera()
    fitter = _fitter(n_peaks=4)
    for reference in (False, True):
        cam._is_reference_mode = lambda: reference
        px, sline = fitter.get_px_sline_from_image(cam.snap().astype(float))
        fs = fitter.fit(px, sline, is_reference_mode=reference)
        assert fs.is_success, f"reference={reference}"
        assert fs.model.startswith("4")
        assert fs.outer_left_peak_center_px is not None
        # geometry: outer orders bracket the inner pair inside the window
        assert (0 < fs.outer_left_peak_center_px < fs.left_peak_center_px
                < fs.right_peak_center_px < fs.outer_right_peak_center_px
                < px[-1])
        # outer orders are the dimmer pair (amplitude ranking depends on it)
        assert fs.outer_left_peak_amplitude < fs.left_peak_amplitude
        assert fs.outer_right_peak_amplitude < fs.right_peak_amplitude


def test_dummy_two_peak_fit_keeps_the_inner_pair():
    cam = _camera()
    f4 = _fitter(n_peaks=4)
    f2 = _fitter(n_peaks=2)
    frame = cam.snap().astype(float)
    px, sline = f4.get_px_sline_from_image(frame)
    r4 = f4.fit(px, sline, is_reference_mode=False)
    r2 = f2.fit(px, sline, is_reference_mode=False)
    assert r4.is_success and r2.is_success
    assert r2.outer_left_peak_center_px is None
    # the two-peak fit lands on the inner main pair, not the outer orders
    assert abs(r2.left_peak_center_px - r4.left_peak_center_px) < 0.5
    assert abs(r2.right_peak_center_px - r4.right_peak_center_px) < 0.5
