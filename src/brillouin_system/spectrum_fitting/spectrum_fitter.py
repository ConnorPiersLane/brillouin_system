import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    find_peaks_sample_config,
    find_peaks_reference_config,
    SlineFromFrameConfig,
    sline_from_frame_config,
    FittingConfigs,
)
from brillouin_system.spectrum_fitting.fit_util import (
    find_peak_locations,
    select_top_two_peaks,
    sort_peaks,
    refine_fitted_spectrum,
)


# === Fitting model functions ===

def _lorentzian_pixel_integrated(x, amp, cen, wid):
    x = np.asarray(x, dtype=float)
    left = x - 0.5
    right = x + 0.5
    return amp * wid * (
        np.arctan((right - cen) / wid) -
        np.arctan((left - cen) / wid)
    )


def _1lorentzian_binned(x, amp, cen, wid, offset):
    return _lorentzian_pixel_integrated(x, amp, cen, wid) + offset


def _2lorentzian_binned(x, amp1, cen1, wid1, amp2, cen2, wid2, offset):
    return (
        _lorentzian_pixel_integrated(x, amp1, cen1, wid1) +
        _lorentzian_pixel_integrated(x, amp2, cen2, wid2) +
        offset
    )


class SpectrumFitter:
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

    def get_px_sline_from_image(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sum the specified vertical rows in the image to produce the Brillouin sline.
        If the row list is invalid or empty, use the full vertical range.

        Parameters:
            frame (np.ndarray): 2D image from the camera

        Returns:
            tuple[np.ndarray, np.ndarray]: Pixel axis and summed/cropped spectrum.
        """
        left_offset = self.sline_config.pixel_offset_left
        right_offset = self.sline_config.pixel_offset_right

        rows = self.sline_config.selected_rows
        height = frame.shape[0]

        if not rows or not all(0 <= r < height for r in rows):
            print("[get_sline_from_image] Warning: Invalid or empty row list — using full image height.")
            rows = list(range(height))

        sline = frame[rows, :].sum(axis=0)
        px = np.arange(sline.shape[0])

        if right_offset > 0:
            sline = sline[left_offset:-right_offset]
            px = px[left_offset:-right_offset]
        else:
            sline = sline[left_offset:]
            px = px[left_offset:]

        return px, sline

    def get_empty_fitting(self, px, sline) -> FittedSpectrum:
        return FittedSpectrum(
            is_success=False,
            x_pixels=px,
            sline=sline,
        )

    def get_total_sline_value(self, sline) -> float:
        """
        Return the total integrated intensity of the sline.

        Parameters:
            sline (np.ndarray): 1D spectrum intensity array

        Returns:
            float: Total summed intensity
        """
        if sline is None:
            return 0.0

        return float(np.sum(sline))

    def fit(self, px: np.ndarray, sline: np.ndarray, is_reference_mode: bool) -> FittedSpectrum:
        config = self.reference_config if is_reference_mode else self.sample_config
        requested_model = config.fitting_model

        if requested_model not in ("lorentzian", "lorentzian_window"):
            raise ValueError(
                f"Unknown model: '{requested_model}'. "
                f"Supported models are 'lorentzian' and 'lorentzian_window'."
            )

        sline = np.clip(sline, 0, None)

        pk_ind, pk_info = find_peak_locations(sline, config=config)
        if len(pk_ind) < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model
            )

        pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
        amp, cen, wid = self._extract_peak_params(pk_ind, pk_info, px, sline)

        n_peaks = len(cen)
        if n_peaks < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model
            )

        x_min = float(np.min(px))
        x_max = float(np.max(px))
        x_span = max(x_max - x_min, 1.0)
        offset = float(np.amin(sline))

        if n_peaks == 1:
            fit_kind = "1lorentzian"
            if requested_model == "lorentzian_window":
                fit_kind = "1lorentzian_window"

            p0 = [amp[0], cen[0], wid[0], offset]
            bounds = (
                [0, x_min, 1e-12, 0],
                [np.inf, x_max, x_span / 2, np.inf]
            )
            model_func = _1lorentzian_binned

        else:
            fit_kind = "2lorentzian"
            if requested_model == "lorentzian_window":
                fit_kind = "2lorentzian_window"

            p0 = [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1], offset]
            bounds = (
                [0, x_min, 1e-12, 0, x_min, 1e-12, 0],
                [np.inf, x_max, x_span / 2, np.inf, x_max, x_span / 2, np.inf]
            )
            model_func = _2lorentzian_binned

        if requested_model == "lorentzian_window":
            beta = self.reference_config.beta if is_reference_mode else self.sample_config.beta

            mask = self._build_window_mask(px, cen, wid, beta=beta)
            px_fit = px[mask]
            sline_fit = sline[mask]

            lo, hi = list(bounds[0]), list(bounds[1])
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)

            if n_peaks == 1:
                lo[1], hi[1] = center_ranges[0]
                lo[2] = max(1e-6, 0.25 * float(wid[0]))
                hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
            else:
                lo[1], hi[1] = center_ranges[0]
                lo[4], hi[4] = center_ranges[1]

                lo[2] = max(1e-6, 0.25 * float(wid[0]))
                hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))

                lo[5] = max(1e-6, 0.25 * float(wid[1]))
                hi[5] = max(lo[5] * 2, 4.0 * float(wid[1]))

            bounds = (lo, hi)
            mask_used = mask
        else:
            px_fit = px
            sline_fit = sline
            mask_used = np.ones_like(px, dtype=bool)

        try:
            popt, _ = curve_fit(
                model_func,
                px_fit,
                sline_fit,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )

            if n_peaks == 2:
                popt[:7] = sort_peaks(popt[:7])

        except Exception as e:
            print(f"[SpectrumFitter] Fit failed: {e}")
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=fit_kind
            )

        return self._build_result(
            px=px,
            sline=sline,
            model_func=model_func,
            popt=popt,
            model=fit_kind,
            mask=mask_used,
        )

    def _extract_peak_params(self, pk_ind, pk_info, px, sline):
        """
        Extract amplitudes, centers, and widths for the detected peaks.

        Returns exactly the peaks that were detected, up to two.
        No synthetic second peak is created.
        """
        pk_ind = np.asarray(pk_ind, dtype=int)

        if len(pk_ind) < 1:
            return np.array([]), np.array([]), np.array([])

        widths_idx = 0.5 * np.asarray(pk_info["widths"], dtype=float)
        heights = np.asarray(pk_info["peak_heights"], dtype=float)

        if "left_ips" in pk_info and "right_ips" in pk_info:
            idx_axis = np.arange(len(px), dtype=float)
            centers_idx = 0.5 * (
                np.asarray(pk_info["left_ips"], dtype=float) +
                np.asarray(pk_info["right_ips"], dtype=float)
            )
            centers = np.interp(centers_idx, idx_axis, px)
        else:
            centers = px[np.clip(pk_ind, 0, len(px) - 1)].astype(float)

        widths = widths_idx * 1.0
        return heights, centers, widths

    def _build_result(self, px, sline, model_func, popt, model: str, mask: np.ndarray) -> FittedSpectrum:
        fitted = model_func(px, *popt)
        x_fit, y_fit = refine_fitted_spectrum(model_func, px, popt, factor=10)

        if len(popt) == 4:
            amp, cen, wid, offset = popt

            return FittedSpectrum(
                is_success=True,
                model=model,
                sline=sline,
                x_pixels=px,
                fitted_spectrum=fitted,
                x_fit_refined=x_fit,
                y_fit_refined=y_fit,
                mask_for_fitting=mask,
                parameters=popt,
                left_peak_center_px=float(cen),
                left_peak_width_px=float(wid),
                left_peak_amplitude=float(amp / 2.0),
                right_peak_center_px=float(cen),
                right_peak_width_px=float(wid),
                right_peak_amplitude=float(amp / 2.0),
                inter_peak_distance=0.0,
                offset=float(offset),
            )

        if len(popt) == 7:
            amp1, cen1, wid1, amp2, cen2, wid2, offset = popt

            return FittedSpectrum(
                is_success=True,
                model=model,
                sline=sline,
                x_pixels=px,
                fitted_spectrum=fitted,
                x_fit_refined=x_fit,
                y_fit_refined=y_fit,
                mask_for_fitting=mask,
                parameters=popt,
                left_peak_center_px=float(cen1),
                left_peak_width_px=float(wid1),
                left_peak_amplitude=float(amp1),
                right_peak_center_px=float(cen2),
                right_peak_width_px=float(wid2),
                right_peak_amplitude=float(amp2),
                inter_peak_distance=abs(cen2 - cen1),
                offset=float(offset),
            )

        return FittedSpectrum(
            is_success=False,
            model=model,
            sline=sline,
            x_pixels=px,
        )

    @staticmethod
    def _build_window_mask(px, centers, widths, beta=4.0):
        mask = np.zeros_like(px, dtype=bool)
        for c, w in zip(centers, widths):
            center_idx = int(np.argmin(np.abs(px - float(c))))
            half = max(int(round(beta * float(w))), 1)
            lo = max(center_idx - half, 0)
            hi = min(center_idx + half + 1, len(px))
            mask[lo:hi] = True
        return mask

    @staticmethod
    def _bounded_center_ranges(px, centers, widths, beta=4.0):
        x_min, x_max = float(np.min(px)), float(np.max(px))
        ranges = []
        for c, w in zip(centers, widths):
            c = float(c)
            half = beta * float(w)
            lo = max(x_min, c - half)
            hi = min(x_max, c + half)
            ranges.append((lo, hi))
        return ranges