import math
from dataclasses import dataclass
import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.analyze_util import get_b_values
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, count_to_electrons

# Thompson's photon term s^2/N is derived for a GAUSSIAN profile. Our peaks are
# Lorentzian and we pass the HWHM in place of s. A Lorentzian's tails carry less
# information about the centre, so its bound is g*sqrt(2/N) -- a factor 2 in
# variance. Verified by Monte-Carlo (2000 fits at matched width and photons) and
# against measured scatter: without this factor the formula reads 0.55-0.77x of
# the real per-frame scatter across water, glycerol and cornea; with it,
# 0.78-1.08x.
LORENTZIAN_CRLB_FACTOR = 2.0


@dataclass
class TheoreticalPeakStdError:
    """ All Values in MHz"""
    left_peak_photons_mhz: float | None =  None
    left_peak_pixelation_mhz: float | None =  None
    left_peak_bg_mhz: float | None =  None
    left_peak_total_mhz: float | None =  None
    right_peak_photons_mhz: float | None =  None
    right_peak_pixelation_mhz: float | None =  None
    right_peak_bg_mhz: float | None =  None
    right_peak_total_mhz: float | None =  None
    # Precision of the peak-distance observable (the one normally reported).
    distance_total_mhz: float | None =  None

@dataclass
class AnalyzedFreqShifts:
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    # Raw fitted width, as the peak lands on the detector: still broadened by
    # the instrument, whatever the lineshape. Unchanged meaning for every model.
    hwhm_left_peak_ghz: float | None
    hwhm_right_peak_ghz: float | None
    # Instrument HWHM from the calibration sidebands, at each peak's own pixel,
    # and the sample linewidth left after subtracting it. The linewidth pair is
    # None unless the fit is pixel-response and the calibration carries a width
    # model (see CalibrationCalculator.sample_linewidth_ghz).
    instrument_hwhm_left_peak_ghz: float | None = None
    instrument_hwhm_right_peak_ghz: float | None = None
    linewidth_left_peak_ghz: float | None = None
    linewidth_right_peak_ghz: float | None = None


@dataclass
class MeasuredStatistics:
    """Means in GHz, std in MHz, covariance in MHz², correlation dimensionless."""
    mean_freq_shift_left_peak_ghz: float | None
    mean_freq_shift_right_peak_ghz: float | None
    mean_freq_shift_peak_distance_ghz: float | None
    std_freq_shift_left_peak_mhz: float | None
    std_freq_shift_right_peak_mhz: float | None
    std_freq_shift_peak_distance_mhz: float | None

    mean_hwhm_left_peak_ghz: float | None
    mean_hwhm_right_peak_ghz: float | None
    std_hwhm_left_peak_mhz: float | None
    std_hwhm_right_peak_mhz: float | None

    cov_freq_left_right: float | None   # MHz²
    corr_freq_left_right: float | None  # dimensionless




class SpectrumAnalyzer:
    def __init__(self, calibration_calculator: CalibrationCalculator):
        self.calibration_calculator = calibration_calculator

    def analyze_spectrum(self, fitting: FittedSpectrum) -> AnalyzedFreqShifts:
        if not fitting.is_success:
            return AnalyzedFreqShifts(
                freq_shift_left_peak_ghz=None,
                freq_shift_right_peak_ghz=None,
                freq_shift_peak_distance_ghz=None,
                hwhm_left_peak_ghz=None,
                hwhm_right_peak_ghz=None,
            )

        calc = self.calibration_calculator
        hwhm_left, hwhm_right = calc.hwhm_ghz(fitting)
        inst_left, inst_right = calc.instrument_hwhm_ghz(
            fitting.left_peak_center_px, fitting.right_peak_center_px)
        width_left, width_right = calc.sample_linewidth_ghz(fitting)

        return AnalyzedFreqShifts(
            freq_shift_left_peak_ghz=calc.freq_left_peak(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz=calc.freq_right_peak(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz=calc.freq_peak_distance(fitting.inter_peak_distance),
            hwhm_left_peak_ghz=hwhm_left,
            hwhm_right_peak_ghz=hwhm_right,
            instrument_hwhm_left_peak_ghz=inst_left,
            instrument_hwhm_right_peak_ghz=inst_right,
            linewidth_left_peak_ghz=width_left,
            linewidth_right_peak_ghz=width_right,
        )


    def theoretical_precision(self, fs: FittedSpectrum,
                              photons: PhotonsCounts,
                              bg_frame_std: np.ndarray | None,
                              preamp_gain: int | float,
                              emccd_gain: int | float,
                              corr_left_right: float = 0.0,
                              ) -> TheoreticalPeakStdError:
        """ See paper: Precise nanometer Localization Analysis for Individual Fluorescent Probes

        THIS IS A LOWER BOUND, NOT A PREDICTION of the pipeline's per-frame
        scatter. It assumes an ideal (maximum-likelihood) estimator and pure
        photon noise from the peak itself. The production pipeline sits a
        measured ~1.45x above it (verified 2026-08-12 by Monte Carlo with
        exactly-known noise; scripts in Data/Calibration_paper_data):
          * x1.14 -- the exact Cramer-Rao bound for the real 7-parameter
            pixel-integrated model with the real noise (read noise + shot
            noise of the stray-light pedestal, both ignored here);
          * x1.28 -- the production prm1 fit is unweighted least squares,
            not maximum likelihood, so it does not reach the bound (the
            plain lorentzian_window fit costs x1.15).
        Measured per-frame scatter matches x1.45 plus the ~0.8 MHz per-peak
        pattern-translation drift, closing the budget with nothing left over.
        Use these numbers as the floor the measurement cannot beat, and scale
        by the factors above when an absolute prediction is needed.

        corr_left_right is the correlation between the two peak-centre errors,
        used only for the distance. Shot noise in the two peaks comes from
        different photons on different pixels, so the default of 0 is the right
        choice for a shot-noise bound. (The ~-0.1 correlation seen in repeated
        measurements is common-mode drift, not photon noise.)
        """
        if not fs.is_success:
            return TheoreticalPeakStdError()


        # All values are in GHz, as this is a distance approx for the spectrometer
        # Lorentzian Profile, approximate std with fwhm. The raw fitted width is
        # the right one here: the bound is set by how wide the peak actually
        # lands on the detector, not by the sample's own linewidth.
        s_l, s_r = self.calibration_calculator.hwhm_ghz(fs)

        a_l = abs(self.calibration_calculator.df_left_peak(px=fs.left_peak_center_px, dpx=1))
        a_r = abs(self.calibration_calculator.df_right_peak(px=fs.right_peak_center_px, dpx=1))

        n_l = photons.left_peak_photons
        n_r = photons.right_peak_photons

        if bg_frame_std is None:
            b_counts_l, b_counts_r = 0, 0
        else:
            b_counts_l, b_counts_r = get_b_values(std_img=bg_frame_std, fit=fs)
        b_l = count_to_electrons(b_counts_l, preamp_gain=preamp_gain, emccd_gain=emccd_gain)
        b_r = count_to_electrons(b_counts_r, preamp_gain=preamp_gain, emccd_gain=emccd_gain)

        dx_l_photons = math.sqrt(  LORENTZIAN_CRLB_FACTOR * s_l ** 2 / n_l        )
        dx_l_pixelation = math.sqrt(  (a_l**2/12) / n_l)
        dx_l_bg =math.sqrt(  4*math.sqrt(math.pi) * s_l**3*b_l**2 / (a_l * n_l**2) )
        dx_l_total =math.sqrt( dx_l_photons**2 + dx_l_pixelation**2 + dx_l_bg**2 )

        dx_r_photons = math.sqrt(  LORENTZIAN_CRLB_FACTOR * s_r ** 2 / n_r        )
        dx_r_pixelation = math.sqrt(  (a_r**2/12) / n_r)
        dx_r_bg =math.sqrt(  4*math.sqrt(math.pi) * s_r**3*b_r**2 / (a_r * n_r**2) )
        dx_r_total =math.sqrt( dx_r_photons**2 + dx_r_pixelation**2 + dx_r_bg**2)

        # Distance observable: d_px = c_right - c_left, so
        #   var(d_px) = var(c_right) + var(c_left) - 2*cov(c_left, c_right).
        # dx_*_total are frequency errors read through each order's own
        # polynomial, so divide by that order's slope to get back to pixels,
        # then apply the distance polynomial's slope.
        a_d = abs(self.calibration_calculator.df_peak_distance(px=fs.inter_peak_distance, dpx=1))
        sigma_c_l_px = dx_l_total / a_l
        sigma_c_r_px = dx_r_total / a_r
        var_d_px = (sigma_c_l_px**2 + sigma_c_r_px**2
                    - 2 * corr_left_right * sigma_c_l_px * sigma_c_r_px)
        dx_d_total = a_d * math.sqrt(max(var_d_px, 0.0))

        return TheoreticalPeakStdError(
            left_peak_photons_mhz=dx_l_photons * 1000,
            left_peak_pixelation_mhz=dx_l_pixelation * 1000,
            left_peak_bg_mhz=dx_l_bg * 1000,
            left_peak_total_mhz=dx_l_total * 1000,
            right_peak_photons_mhz=dx_r_photons * 1000,
            right_peak_pixelation_mhz=dx_r_pixelation * 1000,
            right_peak_bg_mhz=dx_r_bg * 1000,
            right_peak_total_mhz=dx_r_total * 1000,
            distance_total_mhz=dx_d_total * 1000,
        )


def analyze_statistics(
    shifts: list[AnalyzedFreqShifts],
) -> MeasuredStatistics | None:

    if not shifts:
        return None

    def valid_values(values: list[float | None]) -> list[float]:
        return [v for v in values if v is not None]

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def std_mhz_or_none(values: list[float]) -> float | None:
        return float(np.std(values, ddof=1) * 1e3) if len(values) > 1 else None

    def cov_corr_from_pairs(
        pairs: list[tuple[float | None, float | None]]
    ) -> tuple[float | None, float | None]:
        valid_pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(valid_pairs) < 2:
            return None, None

        x = np.array([p[0] for p in valid_pairs], dtype=float)
        y = np.array([p[1] for p in valid_pairs], dtype=float)

        cov_ghz2 = np.cov(x, y, ddof=1)[0, 1]
        cov_mhz2 = float(cov_ghz2 * 1e6)  # GHz² -> MHz²

        std_x_mhz = float(np.std(x, ddof=1) * 1e3)
        std_y_mhz = float(np.std(y, ddof=1) * 1e3)

        corr = None
        if std_x_mhz > 0 and std_y_mhz > 0:
            corr = float(cov_mhz2 / (std_x_mhz * std_y_mhz))

        return cov_mhz2, corr

    # Collect values
    left_poly = valid_values([s.freq_shift_left_peak_ghz for s in shifts])
    right_poly = valid_values([s.freq_shift_right_peak_ghz for s in shifts])
    dist_poly = valid_values([s.freq_shift_peak_distance_ghz for s in shifts])


    hwhm_left = valid_values([s.hwhm_left_peak_ghz for s in shifts])
    hwhm_right = valid_values([s.hwhm_right_peak_ghz for s in shifts])

    # Cov/corr from paired valid values only
    cov_poly, corr_poly = cov_corr_from_pairs([
        (s.freq_shift_left_peak_ghz, s.freq_shift_right_peak_ghz)
        for s in shifts
    ])


    return MeasuredStatistics(
        mean_freq_shift_left_peak_ghz=mean_or_none(left_poly),
        mean_freq_shift_right_peak_ghz=mean_or_none(right_poly),
        mean_freq_shift_peak_distance_ghz=mean_or_none(dist_poly),
        std_freq_shift_left_peak_mhz=std_mhz_or_none(left_poly),
        std_freq_shift_right_peak_mhz=std_mhz_or_none(right_poly),
        std_freq_shift_peak_distance_mhz=std_mhz_or_none(dist_poly),


        mean_hwhm_left_peak_ghz=mean_or_none(hwhm_left),
        mean_hwhm_right_peak_ghz=mean_or_none(hwhm_right),
        std_hwhm_left_peak_mhz=std_mhz_or_none(hwhm_left),
        std_hwhm_right_peak_mhz=std_mhz_or_none(hwhm_right),

        cov_freq_left_right=cov_poly,
        corr_freq_left_right=corr_poly,
    )