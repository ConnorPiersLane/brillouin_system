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

from brillouin_system.spectrum_fitting.voigt_model import (
    _1voigt_binned,
    _2voigt_binned,
)

from brillouin_system.spectrum_fitting.dho_model import (
    ElasticAnchors,
    dho_peak_height,
    epsf_hwhm_px,
    make_2dho_epsf_anchored,
    make_2dho_s2_anchored,
    make_2lorentzian_epsf,
)


# -----------------------------
# Symmetric Lorentzian models
# -----------------------------

def _lorentzian_pixel_integrated(x, amp, cen, wid):
    x = np.asarray(x, dtype=float)
    wid = max(float(wid), 1e-12)
    left = x - 0.5
    right = x + 0.5
    return amp * wid * (
        np.arctan((right - cen) / wid)
        - np.arctan((left - cen) / wid)
    )


def _1lorentzian_binned(x, amp, cen, wid, offset):
    return _lorentzian_pixel_integrated(x, amp, cen, wid) + offset


def _2lorentzian_binned(x, amp1, cen1, wid1, amp2, cen2, wid2, offset):
    return (
        _lorentzian_pixel_integrated(x, amp1, cen1, wid1)
        + _lorentzian_pixel_integrated(x, amp2, cen2, wid2)
        + offset
    )


# -----------------------------
# Asymmetric Lorentzian models
# -----------------------------
# wid_left and wid_right are HWHM-like Lorentzian gamma values.
# For output compatibility, the reported width is their mean.


def _asym_lorentzian_pixel_integrated(x, amp, cen, wid_left, wid_right):
    x = np.asarray(x, dtype=float)

    left = x - 0.5
    right = x + 0.5

    wid_left = max(float(wid_left), 1e-12)
    wid_right = max(float(wid_right), 1e-12)

    y = np.zeros_like(x, dtype=float)

    m_left = right <= cen
    y[m_left] = amp * wid_left * (
        np.arctan((right[m_left] - cen) / wid_left)
        - np.arctan((left[m_left] - cen) / wid_left)
    )

    m_right = left >= cen
    y[m_right] = amp * wid_right * (
        np.arctan((right[m_right] - cen) / wid_right)
        - np.arctan((left[m_right] - cen) / wid_right)
    )

    m_cross = ~(m_left | m_right)
    y[m_cross] = (
        amp * wid_left * np.arctan((cen - left[m_cross]) / wid_left)
        + amp * wid_right * np.arctan((right[m_cross] - cen) / wid_right)
    )

    return y

def _1asym_lorentzian_binned(x, amp, cen, wid_left, wid_right, offset):
    return _asym_lorentzian_pixel_integrated(x, amp, cen, wid_left, wid_right) + offset


def _2asym_lorentzian_binned(
    x,
    amp1, cen1, wid1_left, wid1_right,
    amp2, cen2, wid2_left, wid2_right,
    offset,
):
    return (
        _asym_lorentzian_pixel_integrated(x, amp1, cen1, wid1_left, wid1_right)
        + _asym_lorentzian_pixel_integrated(x, amp2, cen2, wid2_left, wid2_right)
        + offset
    )


def _sort_2asym_lorentzian_params(popt):
    popt = np.asarray(popt, dtype=float)
    if popt[5] < popt[1]:
        return np.array([
            popt[4], popt[5], popt[6], popt[7],
            popt[0], popt[1], popt[2], popt[3],
            popt[8],
        ])
    return popt


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
        if sline is None:
            return 0.0
        return float(np.sum(sline))

    def fit(
        self,
        px: np.ndarray,
        sline: np.ndarray,
        is_reference_mode: bool,
        anchors: ElasticAnchors | None = None,
    ) -> FittedSpectrum:
        config = self.reference_config if is_reference_mode else self.sample_config
        requested_model = config.fitting_model

        if requested_model not in (
            "lorentzian",
            "lorentzian_window",
            "lorentzian_psf",
            "lorentzian_psf_window",
            "asym_lorentzian",
            "asym_lorentzian_window",
            "voigt",
            "voigt_window",
            "dho",
            "dho_window",
            "dho_psf",
            "dho_psf_window",
        ):
            raise ValueError(
                f"Unknown model: '{requested_model}'. "
                f"Supported models are 'lorentzian', 'lorentzian_window', "
                f"'lorentzian_psf', 'lorentzian_psf_window', "
                f"'asym_lorentzian', 'asym_lorentzian_window', "
                f"'voigt', 'voigt_window', 'dho', 'dho_window', "
                f"'dho_psf', and 'dho_psf_window'."
            )

        # Which anchors the model consumes: the pure dho takes the base
        # (main-chain) anchors; the PSF-aware models take the PSF-chain
        # anchors carrying the measured kernels.
        psf_anchors = self._psf_anchors(anchors)

        if requested_model in (
            "lorentzian_psf", "lorentzian_psf_window", "dho_psf", "dho_psf_window"
        ):
            # Forward models through the measured instrument response; only
            # defined when the calibration reconstructed the PSFs.
            if psf_anchors is None:
                raise ValueError(
                    f"The {requested_model} model requires the measured "
                    "instrument PSFs, but the current calibration has no "
                    "usable PSF chain. Run or load a calibration whose PSF "
                    "chain is available, or choose another fitting model."
                )

        if requested_model in ("dho", "dho_window", "dho_psf", "dho_psf_window"):
            # The DHO models are eq. S2 anchored at the elastic (Rayleigh)
            # lines; without the calibration-derived anchors they are not
            # defined. The pure dho needs NOTHING but the anchor positions.
            dho_anchors = (
                psf_anchors
                if requested_model in ("dho_psf", "dho_psf_window")
                else anchors
            )
            if dho_anchors is None:
                raise ValueError(
                    "The DHO model requires elastic anchors from a calibration "
                    "(CalibrationCalculator.elastic_anchors()). Run or load a "
                    "calibration first, or choose another fitting model."
                )
            if not (
                np.isfinite(dho_anchors.rayleigh_left_px)
                and np.isfinite(dho_anchors.rayleigh_right_px)
                and dho_anchors.rayleigh_left_px < dho_anchors.rayleigh_right_px
            ):
                raise ValueError(
                    f"Invalid elastic anchors: left={dho_anchors.rayleigh_left_px}, "
                    f"right={dho_anchors.rayleigh_right_px}."
                )
            # From here on, "anchors" is the set this dho variant consumes
            # (base for the pure S2 dho, PSF-chain for dho_psf);
            # _build_result reads its rayleigh positions.
            anchors = dho_anchors

        # Chain stamp: which calibration chain the fitted pixels pair with.
        # Models inherit it from the anchors they consume (the anchors know
        # which chain produced them); plain models pair with the
        # Lorentzian-centered main chain. The analysis maps the result
        # through this chain (CalibrationCalculator.for_chain).
        if requested_model in (
            "lorentzian_psf", "lorentzian_psf_window", "dho_psf", "dho_psf_window"
        ):
            calibration_chain = psf_anchors.chain or "lorentzian"
        elif requested_model in ("dho", "dho_window"):
            calibration_chain = dho_anchors.chain or "lorentzian"
        else:
            calibration_chain = "lorentzian"

        px = np.asarray(px, dtype=np.float64)
        sline = np.asarray(sline, dtype=np.float64)

        finite_mask = np.isfinite(px) & np.isfinite(sline)
        px = px[finite_mask]
        sline = sline[finite_mask]

        # Keep this if your peak finder expects non-negative data. Remove if you want
        # the offset/background model to handle negative baseline excursions.
        sline = np.clip(sline, 0, None)

        pk_ind, pk_info = find_peak_locations(sline, config=config)
        if len(pk_ind) < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model,
            )

        pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
        amp, cen, wid = self._extract_peak_params(pk_ind, pk_info, px, sline)

        n_peaks = len(cen)
        if n_peaks < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model,
            )

        x_min = float(np.min(px))
        x_max = float(np.max(px))
        x_span = max(x_max - x_min, 1.0)
        offset = float(np.amin(sline))

        r_left = r_right = None
        dho_instrument_hwhm = None

        if requested_model in ("dho", "dho_window", "dho_psf", "dho_psf_window"):
            # The DHO fits both peaks jointly (eq. S2 anchored at each
            # Rayleigh order), so it needs both peaks to be present.
            if n_peaks < 2:
                print(
                    f"[SpectrumFitter] DHO model needs two detected peaks "
                    f"(found {n_peaks}) — returning unsuccessful fit."
                )
                return FittedSpectrum(
                    is_success=False,
                    sline=sline,
                    x_pixels=px,
                    model=requested_model,
                )

            order = np.argsort(cen)
            amp, cen, wid = amp[order], cen[order], wid[order]

            # The elastic anchors must bracket the detected peaks; if they do
            # not (e.g. calibration drift), the fit is not meaningful.
            r_left = float(dho_anchors.rayleigh_left_px)
            r_right = float(dho_anchors.rayleigh_right_px)
            if not (r_left < cen[0] and cen[1] < r_right):
                print(
                    "[SpectrumFitter] Elastic anchors do not bracket the "
                    "detected peaks (calibration drift?) — returning "
                    "unsuccessful fit."
                )
                return FittedSpectrum(
                    is_success=False,
                    sline=sline,
                    x_pixels=px,
                    model=requested_model,
                )

        if requested_model in ("lorentzian_psf", "lorentzian_psf_window"):
            # The PSFs are per-order (left/right kernels differ), so both
            # peaks must be present and assigned to their sides.
            if n_peaks < 2:
                print(
                    f"[SpectrumFitter] lorentzian_psf needs two detected peaks "
                    f"(found {n_peaks}) — returning unsuccessful fit."
                )
                return FittedSpectrum(
                    is_success=False,
                    sline=sline,
                    x_pixels=px,
                    model=requested_model,
                )
            order = np.argsort(cen)
            amp, cen, wid = amp[order], cen[order], wid[order]

        if n_peaks == 1:
            if requested_model in ("lorentzian", "lorentzian_window"):
                fit_kind = "1lorentzian_window" if requested_model == "lorentzian_window" else "1lorentzian"
                p0 = [amp[0], cen[0], wid[0], offset]
                bounds = (
                    [0, x_min, 1e-12, 0],
                    [np.inf, x_max, x_span / 2, np.inf],
                )
                model_func = _1lorentzian_binned

            elif requested_model in ("asym_lorentzian", "asym_lorentzian_window"):
                fit_kind = "1asym_lorentzian_window" if requested_model == "asym_lorentzian_window" else "1asym_lorentzian"
                p0 = [amp[0], cen[0], wid[0], wid[0], offset]
                bounds = (
                    [0, x_min, 1e-12, 1e-12, 0],
                    [np.inf, x_max, x_span / 2, x_span / 2, np.inf],
                )
                model_func = _1asym_lorentzian_binned

            elif requested_model in ("voigt", "voigt_window"):
                fit_kind = "1voigt_window" if requested_model == "voigt_window" else "1voigt"
                p0 = [amp[0], cen[0], wid[0], 0.25, offset]
                bounds = (
                    [0, x_min, 0.03, 0.0, -np.inf],
                    [np.inf, x_max, x_span / 2, 5.0, np.inf],
                )
                model_func = _1voigt_binned

        else:
            if requested_model in ("lorentzian", "lorentzian_window"):
                fit_kind = "2lorentzian_window" if requested_model == "lorentzian_window" else "2lorentzian"
                p0 = [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1], offset]
                bounds = (
                    [0, x_min, 1e-12, 0, x_min, 1e-12, 0],
                    [np.inf, x_max, x_span / 2, np.inf, x_max, x_span / 2, np.inf],
                )
                model_func = _2lorentzian_binned

            elif requested_model in ("asym_lorentzian", "asym_lorentzian_window"):
                fit_kind = "2asym_lorentzian_window" if requested_model == "asym_lorentzian_window" else "2asym_lorentzian"
                p0 = [
                    amp[0], cen[0], wid[0], wid[0],
                    amp[1], cen[1], wid[1], wid[1],
                    offset,
                ]
                bounds = (
                    [
                        0, x_min, 1e-12, 1e-12,
                        0, x_min, 1e-12, 1e-12,
                        0,
                    ],
                    [
                        np.inf, x_max, x_span / 2, x_span / 2,
                        np.inf, x_max, x_span / 2, x_span / 2,
                        np.inf,
                    ],
                )
                model_func = _2asym_lorentzian_binned

            elif requested_model in ("voigt", "voigt_window"):
                fit_kind = "2voigt_window" if requested_model == "voigt_window" else "2voigt"
                p0 = [
                    amp[0], cen[0], wid[0], 0.25,
                    amp[1], cen[1], wid[1], 0.25,
                    offset,
                ]
                bounds = (
                    [
                        0, x_min, 0.03, 0.0,
                        0, x_min, 0.03, 0.0,
                        -np.inf,
                    ],
                    [
                        np.inf, x_max, x_span / 2, 5.0,
                        np.inf, x_max, x_span / 2, 5.0,
                        np.inf,
                    ],
                )
                model_func = _2voigt_binned

            elif requested_model in ("lorentzian_psf", "lorentzian_psf_window"):
                # Lorentzian material cores forward-modeled through the
                # measured per-order PSFs: fitted centers are unbiased under a
                # non-Lorentzian instrument response, fitted gammas are the
                # MATERIAL HWHMs. cen is sorted ascending here (see block
                # above) so peak 1 gets the left-order kernel.
                fit_kind = "2lorentzian_psf_window" if requested_model == "lorentzian_psf_window" else "2lorentzian_psf"

                gi_left = max(epsf_hwhm_px(psf_anchors.psf_left, psf_anchors.psf_grid_step_px), 1e-3)
                gi_right = max(epsf_hwhm_px(psf_anchors.psf_right, psf_anchors.psf_grid_step_px), 1e-3)

                # wid from find_peaks is the observed HWHM; material ~ observed
                # minus instrument. Observed height ~ amp * g / (g + gi).
                g1_0 = min(max(float(wid[0]) - gi_left, 5e-2), x_span / 2)
                g2_0 = min(max(float(wid[1]) - gi_right, 5e-2), x_span / 2)
                amp1_0 = amp[0] * (g1_0 + gi_left) / g1_0
                amp2_0 = amp[1] * (g2_0 + gi_right) / g2_0

                p0 = [amp1_0, cen[0], g1_0, amp2_0, cen[1], g2_0, offset]
                bounds = (
                    [0, x_min, 1e-3, 0, x_min, 1e-3, 0],
                    [np.inf, x_max, x_span / 2, np.inf, x_max, x_span / 2, np.inf],
                )
                model_func = make_2lorentzian_epsf(
                    psf_anchors.psf_left, psf_anchors.psf_right, psf_anchors.psf_grid_step_px
                )
                dho_instrument_hwhm = (gi_left, gi_right)

            elif requested_model in ("dho", "dho_window", "dho_psf", "dho_psf_window"):
                is_psf_dho = requested_model in ("dho_psf", "dho_psf_window")
                if is_psf_dho:
                    # Eq. S2 convolved with the measured per-order ePSF:
                    # omega/gamma are MATERIAL values (instrument deconvolved,
                    # handles skewed / non-Lorentzian responses).
                    fit_kind = "2dho_anchored_psf_window" if requested_model == "dho_psf_window" else "2dho_anchored_psf"
                    gi_left = max(epsf_hwhm_px(psf_anchors.psf_left, psf_anchors.psf_grid_step_px), 1e-3)
                    gi_right = max(epsf_hwhm_px(psf_anchors.psf_right, psf_anchors.psf_grid_step_px), 1e-3)
                    dho_instrument_hwhm = (gi_left, gi_right)
                else:
                    # PURE eq. S2 from the paper: bare DHO anchored at the
                    # elastic lines, no instrument model. The only calibration
                    # input is the anchor positions; the fitted gammas are the
                    # TOTAL observed dampings (material + instrument mixed).
                    fit_kind = "2dho_s2_window" if requested_model == "dho_window" else "2dho_s2"
                    gi_left = gi_right = 0.0

                # cen is sorted ascending here (see dho block above).
                d = max(float(cen[1] - cen[0]), 1.0)

                # wid from find_peaks is a total HWHM-like value; the gamma
                # guess subtracts the instrument HWHM (zero for the pure S2).
                # The two peaks' gammas are independent.
                def _g_guess(w, gi):
                    return min(max(2.0 * (float(w) - gi), 1e-2), 2.0 * d)
                g1_0 = _g_guess(wid[0], gi_left)
                g2_0 = _g_guess(wid[1], gi_right)

                fsr_px = r_right - r_left
                omega1_0 = max(float(cen[0]) - r_left, 1.0)
                omega2_0 = max(r_right - float(cen[1]), 1.0)
                height_unit1 = dho_peak_height(omega1_0, g1_0)
                height_unit2 = dho_peak_height(omega2_0, g2_0)

                p0 = [
                    amp[0] / height_unit1, omega1_0, g1_0,
                    amp[1] / height_unit2, omega2_0, g2_0,
                    offset,
                ]
                bounds = (
                    [0, 1e-2, 1e-3, 0, 1e-2, 1e-3, 0],
                    [np.inf, fsr_px, 2.0 * d, np.inf, fsr_px, 2.0 * d, np.inf],
                )
                if is_psf_dho:
                    model_func = make_2dho_epsf_anchored(
                        r_left, r_right,
                        psf_anchors.psf_left, psf_anchors.psf_right,
                        psf_anchors.psf_grid_step_px,
                    )
                else:
                    model_func = make_2dho_s2_anchored(r_left, r_right)

        if requested_model in ("lorentzian_window", "asym_lorentzian_window", "voigt_window"):
            beta = self.reference_config.beta if is_reference_mode else self.sample_config.beta

            mask = self._build_window_mask(px, cen, wid, beta=beta)
            px_fit = px[mask]
            sline_fit = sline[mask]

            lo, hi = list(bounds[0]), list(bounds[1])
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)

            if n_peaks == 1:
                lo[1], hi[1] = center_ranges[0]

                if requested_model == "lorentzian_window":
                    lo[2] = max(1e-6, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))

                elif requested_model == "asym_lorentzian_window":
                    lo[2] = max(1e-6, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
                    lo[3] = max(1e-6, 0.25 * float(wid[0]))
                    hi[3] = max(lo[3] * 2, 4.0 * float(wid[0]))

                elif requested_model == "voigt_window":
                    lo[2] = max(0.03, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
                    lo[3] = 0.0
                    hi[3] = 5.0

            else:
                lo[1], hi[1] = center_ranges[0]

                if requested_model == "lorentzian_window":
                    lo[4], hi[4] = center_ranges[1]

                    lo[2] = max(1e-6, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))

                    lo[5] = max(1e-6, 0.25 * float(wid[1]))
                    hi[5] = max(lo[5] * 2, 4.0 * float(wid[1]))

                elif requested_model == "asym_lorentzian_window":
                    lo[5], hi[5] = center_ranges[1]

                    lo[2] = max(1e-6, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
                    lo[3] = max(1e-6, 0.25 * float(wid[0]))
                    hi[3] = max(lo[3] * 2, 4.0 * float(wid[0]))

                    lo[6] = max(1e-6, 0.25 * float(wid[1]))
                    hi[6] = max(lo[6] * 2, 4.0 * float(wid[1]))
                    lo[7] = max(1e-6, 0.25 * float(wid[1]))
                    hi[7] = max(lo[7] * 2, 4.0 * float(wid[1]))

                elif requested_model == "voigt_window":
                    lo[5], hi[5] = center_ranges[1]

                    lo[2] = max(0.03, 0.25 * float(wid[0]))
                    hi[2] = max(lo[2] * 2, 4.0 * float(wid[0]))
                    lo[3] = 0.0
                    hi[3] = 5.0

                    lo[6] = max(0.03, 0.25 * float(wid[1]))
                    hi[6] = max(lo[6] * 2, 4.0 * float(wid[1]))
                    lo[7] = 0.0
                    hi[7] = 5.0

            bounds = (lo, hi)
            mask_used = mask

        elif requested_model == "lorentzian_psf_window":
            beta = self.reference_config.beta if is_reference_mode else self.sample_config.beta

            mask = self._build_window_mask(px, cen, wid, beta=beta)
            px_fit = px[mask]
            sline_fit = sline[mask]

            lo, hi = list(bounds[0]), list(bounds[1])
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)

            # Param order: [amp1, cen1, gamma1, amp2, cen2, gamma2, offset]
            lo[1], hi[1] = center_ranges[0]
            lo[4], hi[4] = center_ranges[1]

            # Material widths can be much smaller than the observed widths
            # (instrument-dominated peaks), so keep the lower bounds loose.
            lo[2] = 1e-3
            hi[2] = max(4.0 * float(wid[0]), 0.5)
            lo[5] = 1e-3
            hi[5] = max(4.0 * float(wid[1]), 0.5)

            bounds = (lo, hi)
            mask_used = mask

        elif requested_model in ("dho_window", "dho_psf_window"):
            beta = self.reference_config.beta if is_reference_mode else self.sample_config.beta

            mask = self._build_window_mask(px, cen, wid, beta=beta)
            px_fit = px[mask]
            sline_fit = sline[mask]

            lo, hi = list(bounds[0]), list(bounds[1])
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)

            # Param order: [amp1, omega1, gamma1, amp2, omega2, gamma2, offset]
            # Translate the pixel center ranges into omega ranges relative
            # to the fixed anchors (left peak sits at r_left + u_pk, right
            # peak at r_right - u_pk).
            lo[1] = max(1e-2, center_ranges[0][0] - r_left)
            hi[1] = max(lo[1] * 2, center_ranges[0][1] - r_left)
            lo[4] = max(1e-2, r_right - center_ranges[1][1])
            hi[4] = max(lo[4] * 2, r_right - center_ranges[1][0])

            lo[2] = max(1e-3, 0.5 * float(wid[0]))
            hi[2] = max(lo[2] * 2, 8.0 * float(wid[0]))
            lo[5] = max(1e-3, 0.5 * float(wid[1]))
            hi[5] = max(lo[5] * 2, 8.0 * float(wid[1]))

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
                method="trf",
                maxfev=50000,
            )

            if n_peaks == 2:
                if requested_model in ("lorentzian", "lorentzian_window"):
                    popt[:7] = sort_peaks(popt[:7])

                elif requested_model in ("asym_lorentzian", "asym_lorentzian_window"):
                    popt = _sort_2asym_lorentzian_params(popt)

                elif requested_model in ("voigt", "voigt_window"):
                    if popt[5] < popt[1]:
                        popt = np.array([
                            popt[4], popt[5], popt[6], popt[7],
                            popt[0], popt[1], popt[2], popt[3],
                            popt[8],
                        ])

                # dho: anchored params are side-bound by construction
                # (omega1 belongs to the left anchor), no swap needed.

        except Exception as e:
            print(f"[SpectrumFitter] Fit failed: {e}")
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=fit_kind,
            )

        result = self._build_result(
            px=px,
            sline=sline,
            model_func=model_func,
            popt=popt,
            model=fit_kind,
            mask=mask_used,
            anchors=anchors,
            instrument_hwhm=dho_instrument_hwhm,
        )
        result.calibration_chain = calibration_chain
        return result

    @staticmethod
    def _psf_anchors(anchors: ElasticAnchors | None) -> ElasticAnchors | None:
        """
        The anchors carrying the measured ePSFs, for the PSF-aware models:
        the nested PSF-chain anchors when present, otherwise the base anchors
        if they carry PSFs themselves (legacy single-chain calibrations that
        stored the PSFs at the top level), otherwise None.
        """
        if anchors is None:
            return None
        variant = getattr(anchors, "psf_variant", None)
        for cand in (variant, anchors):
            if (
                cand is not None
                and cand.psf_left is not None
                and cand.psf_right is not None
                and cand.psf_grid_step_px
            ):
                return cand
        return None

    def _extract_peak_params(self, pk_ind, pk_info, px, sline):
        pk_ind = np.asarray(pk_ind, dtype=int)

        if len(pk_ind) < 1:
            return np.array([]), np.array([]), np.array([])

        widths_idx = 0.5 * np.asarray(pk_info["widths"], dtype=float)
        heights = np.asarray(pk_info["peak_heights"], dtype=float)

        if "left_ips" in pk_info and "right_ips" in pk_info:
            idx_axis = np.arange(len(px), dtype=float)
            centers_idx = 0.5 * (
                np.asarray(pk_info["left_ips"], dtype=float)
                + np.asarray(pk_info["right_ips"], dtype=float)
            )
            centers = np.interp(centers_idx, idx_axis, px)
        else:
            centers = px[np.clip(pk_ind, 0, len(px) - 1)].astype(float)

        widths = widths_idx * 1.0
        return heights, centers, widths

    def _build_result(
        self,
        px,
        sline,
        model_func,
        popt,
        model: str,
        mask: np.ndarray,
        anchors: ElasticAnchors | None = None,
        instrument_hwhm: tuple | None = None,
    ) -> FittedSpectrum:
        fitted = model_func(px, *popt)
        x_fit, y_fit = refine_fitted_spectrum(model_func, px, popt, factor=10)

        if len(popt) == 7 and model.startswith("2lorentzian_psf"):
            amp1, cen1, gamma1, amp2, cen2, gamma2, offset = popt
            gi_left, gi_right = instrument_hwhm

            # Option-2 width semantics (same as the DHO): the classic width
            # fields carry the TOTAL observed HWHM (photon counts need the
            # real peak area), the material HWHM is reported separately.
            hwhm_mat1, hwhm_mat2 = float(gamma1), float(gamma2)
            hwhm_tot1 = hwhm_mat1 + float(gi_left)
            hwhm_tot2 = hwhm_mat2 + float(gi_right)
            height_obs1 = float(amp1) * hwhm_mat1 / hwhm_tot1 if hwhm_tot1 > 0 else float(amp1)
            height_obs2 = float(amp2) * hwhm_mat2 / hwhm_tot2 if hwhm_tot2 > 0 else float(amp2)

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
                left_peak_width_px=hwhm_tot1,
                left_peak_amplitude=height_obs1,
                right_peak_center_px=float(cen2),
                right_peak_width_px=hwhm_tot2,
                right_peak_amplitude=height_obs2,
                inter_peak_distance=float(abs(cen2 - cen1)),
                offset=float(offset),
                material_hwhm_left_px=hwhm_mat1,
                material_hwhm_right_px=hwhm_mat2,
            )

        if len(popt) == 7 and model.startswith("2dho"):
            amp1, omega1, g1, amp2, omega2, g2, offset = popt

            r_left = float(anchors.rayleigh_left_px)
            r_right = float(anchors.rayleigh_right_px)

            # The reported peak centers are the RESONANCE positions
            # (rayleigh +/- omega), so the downstream calibration converts them
            # directly into the damping-corrected Brillouin shift.
            cen1 = r_left + float(omega1)
            cen2 = r_right - float(omega2)

            height1 = float(amp1) * dho_peak_height(float(omega1), float(g1))
            height2 = float(amp2) * dho_peak_height(float(omega2), float(g2))

            if instrument_hwhm is None:
                # Pure eq. S2 ("2dho_s2*"): no instrument model, so gamma is
                # the TOTAL observed damping. No material/instrument split.
                hwhm_mat1 = hwhm_mat2 = None
                hwhm_tot1 = 0.5 * float(g1)
                hwhm_tot2 = 0.5 * float(g2)
                height_obs1 = height1
                height_obs2 = height2
            else:
                # ePSF-convolved ("2dho_anchored_psf*"): gamma is MATERIAL.
                # Width fields (option 2): report the TOTAL, observed HWHM so
                # photon counts see the real peak area (convolution conserves
                # area). The material HWHM (for the loss modulus) is reported
                # separately. In the near-Lorentzian limit total HWHM =
                # material HWHM + instrument HWHM, and the convolved peak
                # height follows from area conservation.
                gi_left, gi_right = instrument_hwhm
                hwhm_mat1 = 0.5 * float(g1)
                hwhm_mat2 = 0.5 * float(g2)
                hwhm_tot1 = hwhm_mat1 + float(gi_left)
                hwhm_tot2 = hwhm_mat2 + float(gi_right)
                height_obs1 = height1 * (hwhm_mat1 / hwhm_tot1) if hwhm_tot1 > 0 else height1
                height_obs2 = height2 * (hwhm_mat2 / hwhm_tot2) if hwhm_tot2 > 0 else height2

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
                left_peak_width_px=hwhm_tot1,
                left_peak_amplitude=height_obs1,
                right_peak_center_px=float(cen2),
                right_peak_width_px=hwhm_tot2,
                right_peak_amplitude=height_obs2,
                inter_peak_distance=float(cen2 - cen1),
                offset=float(offset),
                rayleigh_left_px=r_left,
                rayleigh_right_px=r_right,
                material_hwhm_left_px=hwhm_mat1,
                material_hwhm_right_px=hwhm_mat2,
            )

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

        if len(popt) == 5 and model.startswith("1asym_lorentzian"):
            amp, cen, wid_left, wid_right, offset = popt
            wid_mean = 0.5 * (float(wid_left) + float(wid_right))

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
                left_peak_width_px=wid_mean,
                left_peak_amplitude=float(amp / 2.0),
                right_peak_center_px=float(cen),
                right_peak_width_px=wid_mean,
                right_peak_amplitude=float(amp / 2.0),
                inter_peak_distance=0.0,
                offset=float(offset),
            )

        if len(popt) == 5:
            amp, cen, gamma, sigma_psf, offset = popt

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
                left_peak_width_px=float(gamma),
                left_peak_amplitude=float(amp / 2.0),
                right_peak_center_px=float(cen),
                right_peak_width_px=float(gamma),
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

        if len(popt) == 9 and model.startswith("2asym_lorentzian"):
            (
                amp1, cen1, wid1_left, wid1_right,
                amp2, cen2, wid2_left, wid2_right,
                offset,
            ) = popt
            wid1_mean = 0.5 * (float(wid1_left) + float(wid1_right))
            wid2_mean = 0.5 * (float(wid2_left) + float(wid2_right))

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
                left_peak_width_px=wid1_mean,
                left_peak_amplitude=float(amp1),
                right_peak_center_px=float(cen2),
                right_peak_width_px=wid2_mean,
                right_peak_amplitude=float(amp2),
                inter_peak_distance=abs(cen2 - cen1),
                offset=float(offset),
            )

        if len(popt) == 9:
            (
                amp1, cen1, gamma1, sigma_psf1,
                amp2, cen2, gamma2, sigma_psf2,
                offset,
            ) = popt

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
                left_peak_width_px=float(gamma1),
                left_peak_amplitude=float(amp1),
                right_peak_center_px=float(cen2),
                right_peak_width_px=float(gamma2),
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
