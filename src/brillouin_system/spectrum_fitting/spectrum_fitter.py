
import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig, \
    find_peaks_sample_config, find_peaks_reference_config, SlineFromFrameConfig, sline_from_frame_config, FittingConfigs
from brillouin_system.spectrum_fitting.fit_util import (
    find_peak_locations,
    select_top_two_peaks,
    sort_peaks,
    refine_fitted_spectrum,
)



# === Fitting model functions ===

def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def _2Lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2, offset):
    return (amp1 * wid1**2 / ((x - cen1)**2 + wid1**2)) + \
           (amp2 * wid2**2 / ((x - cen2)**2 + wid2**2)) + offset


# === Fitting Engine ===
def get_empty_fitting(sline):
    return FittedSpectrum(
        is_success=False,
        x_pixels=np.arange(sline.shape[0]),
        sline=sline,
    )


class SpectrumFitter:

    @staticmethod
    def get_empty_fitting(sline):
        return FittedSpectrum(
            is_success=False,
            x_pixels=np.arange(sline.shape[0]),
            sline=sline,
        )

    def __init__(self):
        self.sline_config: SlineFromFrameConfig = sline_from_frame_config.get()
        self.sample_config: FindPeaksConfig = find_peaks_sample_config.get()
        self.reference_config: FindPeaksConfig = find_peaks_reference_config.get()

    def update_configs(self, configs: FittingConfigs):
        self.update_sline_config(configs.sline_config)
        self.update_sample_config(configs.sample_config)
        self.update_reference_config(configs.reference_config)


    def update_sline_config(self, sline_config: SlineFromFrameConfig):
        if not isinstance(sline_config, SlineFromFrameConfig):
            raise TypeError("sline_config must be a SlineFromFrame instance.")
        self.sline_config = sline_config

    def update_sample_config(self, sample_config: FindPeaksConfig):
        if not isinstance(sample_config, FindPeaksConfig):
            raise TypeError("sample_config must be a FindPeaksConfig instance.")
        self.sample_config = sample_config

    def update_reference_config(self, reference_config: FindPeaksConfig):
        if not isinstance(reference_config, FindPeaksConfig):
            raise TypeError("reference_config must be a FindPeaksConfig instance.")
        self.reference_config = reference_config

    def get_sline_from_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Sum the specified vertical rows in the image to produce the Brillouin sline.
        If the row list is invalid or empty, use the full vertical range.

        Parameters:
            frame (np.ndarray): 2D image from the camera

        Returns:
            np.ndarray: The summed and cropped spectrum.
        """
        rows = self.sline_config.selected_rows
        height = frame.shape[0]

        if not rows or not all(0 <= r < height for r in rows):
            print("[get_sline_from_image] Warning: Invalid or empty row list — using full image height.")
            rows = list(range(height))

        sline = frame[rows, :].sum(axis=0)

        left_offset = self.sline_config.pixel_offset_left
        right_offset = self.sline_config.pixel_offset_right

        if right_offset > 0:
            sline = sline[left_offset:-right_offset]
        else:
            sline = sline[left_offset:]

        return sline

    def fit(self, sline: np.ndarray, is_reference_mode: bool) -> FittedSpectrum:

        if is_reference_mode:
            model = self.reference_config.fitting_model
        else:
            model = self.sample_config.fitting_model

        config = self.reference_config if is_reference_mode else self.sample_config
        px = np.arange(sline.shape[0])
        sline = np.clip(sline, 0, None)

        pk_ind, pk_info = find_peak_locations(sline, config=config)
        if len(pk_ind) < 1:
            return FittedSpectrum(is_success=False, sline=sline, x_pixels=px, model=model)

        pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
        amp, cen, wid = self._extract_peak_params(pk_ind, pk_info, sline)

        if model == "lorentzian":
            offset = np.amin(sline)
            p0 = [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1], offset]
            bounds = (
                [0, 0, 0, 0, 0, 0, 0],
                [np.inf, max(px), max(px) / 2, np.inf, max(px), max(px) / 2, np.inf]
            )
            model_func = _2Lorentzian

        elif model == "lorentzian_quad_bg":
            p0_bg = [0.0, 0.0, 0.0]
            p0 = [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1], 0.0, *p0_bg]
            bounds = (
                [0, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf],
                [np.inf, max(px), max(px) / 2, np.inf, max(px), max(px) / 2, np.inf, np.inf, np.inf, np.inf]
            )
            def model_func(x, *params):
                return _2Lorentzian(x, *params[:7]) + _quadratic(x, *params[7:])
        elif model == 'lorentzian_window':
            # Same underlying model function as "lorentzian"
            offset = np.amin(sline)
            p0 = [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1], offset]
            bounds = (
                [0, 0, 0, 0, 0, 0, 0],
                [np.inf, max(px), max(px) / 2, np.inf, max(px), max(px) / 2, np.inf]
            )
            model_func = _2Lorentzian
            # Special handling comes below (masking/windowing)
        else:
            raise ValueError(f"Unknown model: '{model}'")

        if model == "lorentzian_window":
            # Build window mask around detected centers

            if is_reference_mode:
                beta = self.reference_config.beta
            else:
                beta = self.sample_config.beta
            mask = self._build_window_mask(px, cen, wid, beta=beta)

            px_fit = px[mask]
            sline_fit = sline[mask]

            # Optionally tighten bounds on centers/widths
            lo, hi = list(bounds[0]), list(bounds[1])
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)

            lo[1], hi[1] = center_ranges[0]
            lo[4], hi[4] = center_ranges[1]

            # constrain widths reasonably
            lo[2] = max(1e-6, 0.25 * float(wid[0]))
            hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
            lo[5] = max(1e-6, 0.25 * float(wid[1]))
            hi[5] = max(lo[5] * 2, 4.0 * float(wid[1]))

            bounds = (lo, hi)

            # Save these for later
            x_masked, y_masked = px_fit, sline_fit
        else:
            px_fit = px
            sline_fit = sline
            x_masked, y_masked = px, sline

        try:
            popt, _ = curve_fit(model_func, px_fit, sline_fit, p0=p0, bounds=bounds, maxfev=10000)
            popt[:7] = sort_peaks(popt[:7])
        except Exception as e:
            print(f"[SpectrumFitter] Fit failed: {e}")
            return FittedSpectrum(is_success=False, sline=sline, x_pixels=px, model=model)

        return self._build_result(px, sline, model_func, popt, model=model, x_masked=x_masked, y_masked=y_masked)

    def _extract_peak_params(self, pk_ind, pk_info, sline):
        if len(pk_ind) == 1:
            offset = max(int(0.02 * len(sline)), 1)
            pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
            widths = np.array([pk_info['widths'][0]] * 2)
            heights = np.array([pk_info['peak_heights'][0] / 2] * 2)
        else:
            widths = 0.5 * pk_info['widths']
            heights = np.pi * widths * pk_info['peak_heights']
        return heights, pk_ind, widths



    def _build_result(self, px, sline, model_func, popt, model: str, x_masked, y_masked) -> FittedSpectrum:
        amp1, cen1, wid1, amp2, cen2, wid2 = popt[:6]
        fitted = model_func(px, *popt)

        x_fit, y_fit = refine_fitted_spectrum(model_func, px, popt, factor=10)

        return FittedSpectrum(
            is_success=True,
            model=model,
            sline=sline,
            x_pixels=px,
            fitted_spectrum=fitted,
            x_fit_refined=x_fit,
            y_fit_refined=y_fit,
            x_masked=x_masked,
            y_masked=y_masked,
            parameters=popt,
            left_peak_center_px=float(cen1),
            left_peak_width_px=float(wid1),
            left_peak_amplitude=float(amp1),
            right_peak_center_px=float(cen2),
            right_peak_width_px=float(wid2),
            right_peak_amplitude=float(amp2),
            inter_peak_distance=abs(cen2 - cen1),
        )

    def _build_window_mask(self, px, centers, widths, beta=4.0):
        mask = np.zeros_like(px, dtype=bool)
        for c, w in zip(centers, widths):
            half = max(int(beta * float(w)), 1)
            lo = max(int(c) - half, 0)
            hi = min(int(c) + half + 1, px[-1] + 1)
            mask[lo:hi] = True
        return mask

    def _bounded_center_ranges(self, px, centers, widths, beta=4.0):
        ranges = []
        for c, w in zip(centers, widths):
            half = beta * float(w)
            lo = max(0.0, float(c) - half)
            hi = min(float(px[-1]), float(c) + half)
            ranges.append((lo, hi))
        return ranges
