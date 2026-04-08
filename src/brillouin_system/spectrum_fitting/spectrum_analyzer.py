import math
from dataclasses import dataclass
import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.analyze_util import get_b_values
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, count_to_electrons


@dataclass
class TheoreticalPeakStdError:
    """ All Values in MHz"""
    left_peak_photons_mhz: float
    left_peak_pixelation_mhz: float
    left_peak_bg_mhz: float
    left_peak_total_mhz: float
    right_peak_photons_mhz: float
    right_peak_pixelation_mhz: float
    right_peak_bg_mhz: float
    right_peak_total_mhz: float

@dataclass
class AnalyzedFreqShifts:
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    hwhm_left_peak_ghz: float | None
    hwhm_right_peak_ghz: float | None


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

        return AnalyzedFreqShifts(
            freq_shift_left_peak_ghz=self.calibration_calculator.freq_left_peak(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz=self.calibration_calculator.freq_right_peak(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz=self.calibration_calculator.freq_peak_distance(fitting.inter_peak_distance),
            hwhm_left_peak_ghz=float(
                abs(
                    self.calibration_calculator.df_left_peak(
                        fitting.left_peak_center_px,
                        fitting.left_peak_width_px
                    )
                )
            ),
            hwhm_right_peak_ghz=float(
                abs(
                    self.calibration_calculator.df_right_peak(
                        fitting.right_peak_center_px,
                        fitting.right_peak_width_px
                    )
                )
            ),
            freq_shift_left_peak_ghz_interp=self.calibration_calculator.freq_left_peak_interp(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz_interp=self.calibration_calculator.freq_right_peak_interp(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz_interp=self.calibration_calculator.freq_peak_distance_interp(fitting.inter_peak_distance),
        )


    def theoretical_precision(self, fs: FittedSpectrum,
                              photons: PhotonsCounts,
                              bg_frame_std: np.ndarray | None,
                              preamp_gain: int | float,
                              emccd_gain: int | float,
                              ) -> TheoreticalPeakStdError:
        """ See paper: Precise nanometer Localization Analysis for Individual Fluorescent Probes """

        analyzed_spec = self.analyze_spectrum(fitting=fs)

        # All values are in GHz, as this is a distance approx for the spectrometer
        # Lorentzian Profile, approximate std with fwhm
        s_l, s_r = analyzed_spec.hwhm_left_peak_ghz, analyzed_spec.hwhm_right_peak_ghz

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

        dx_l_photons = math.sqrt(            s_l ** 2 / n_l        )
        dx_l_pixelation = math.sqrt(  (a_l**2/12) / n_l)
        dx_l_bg =math.sqrt(  4*math.sqrt(math.pi) * s_l**3*b_l**2 / (a_l * n_l**2) )
        dx_l_total =math.sqrt( dx_l_photons**2 + dx_l_pixelation**2 + dx_l_bg**2 )

        dx_r_photons = math.sqrt(            s_r ** 2 / n_r        )
        dx_r_pixelation = math.sqrt(  (a_r**2/12) / n_r)
        dx_r_bg =math.sqrt(  4*math.sqrt(math.pi) * s_r**3*b_r**2 / (a_r * n_r**2) )
        dx_r_total =math.sqrt( dx_r_photons**2 + dx_r_pixelation**2 + dx_r_bg**2)


        return TheoreticalPeakStdError(
            left_peak_photons_mhz=dx_l_photons * 1000,
            left_peak_pixelation_mhz=dx_l_pixelation * 1000,
            left_peak_bg_mhz=dx_l_bg * 1000,
            left_peak_total_mhz=dx_l_total * 1000,
            right_peak_photons_mhz=dx_r_photons * 1000,
            right_peak_pixelation_mhz=dx_r_pixelation * 1000,
            right_peak_bg_mhz=dx_r_bg * 1000,
            right_peak_total_mhz=dx_r_total * 1000,
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

    left_interp = valid_values([s.freq_shift_left_peak_ghz_interp for s in shifts])
    right_interp = valid_values([s.freq_shift_right_peak_ghz_interp for s in shifts])
    dist_interp = valid_values([s.freq_shift_peak_distance_ghz_interp for s in shifts])

    hwhm_left = valid_values([s.hwhm_left_peak_ghz for s in shifts])
    hwhm_right = valid_values([s.hwhm_right_peak_ghz for s in shifts])

    # Cov/corr from paired valid values only
    cov_poly, corr_poly = cov_corr_from_pairs([
        (s.freq_shift_left_peak_ghz, s.freq_shift_right_peak_ghz)
        for s in shifts
    ])

    cov_interp, corr_interp = cov_corr_from_pairs([
        (s.freq_shift_left_peak_ghz_interp, s.freq_shift_right_peak_ghz_interp)
        for s in shifts
    ])

    return MeasuredStatistics(
        mean_freq_shift_left_peak_ghz=mean_or_none(left_poly),
        mean_freq_shift_right_peak_ghz=mean_or_none(right_poly),
        mean_freq_shift_peak_distance_ghz=mean_or_none(dist_poly),
        std_freq_shift_left_peak_mhz=std_mhz_or_none(left_poly),
        std_freq_shift_right_peak_mhz=std_mhz_or_none(right_poly),
        std_freq_shift_peak_distance_mhz=std_mhz_or_none(dist_poly),

        mean_freq_shift_left_peak_ghz_interp=mean_or_none(left_interp),
        mean_freq_shift_right_peak_ghz_interp=mean_or_none(right_interp),
        mean_freq_shift_peak_distance_ghz_interp=mean_or_none(dist_interp),
        std_freq_shift_left_peak_mhz_interp=std_mhz_or_none(left_interp),
        std_freq_shift_right_peak_mhz_interp=std_mhz_or_none(right_interp),
        std_freq_shift_peak_distance_mhz_interp=std_mhz_or_none(dist_interp),

        mean_hwhm_left_peak_ghz=mean_or_none(hwhm_left),
        mean_hwhm_right_peak_ghz=mean_or_none(hwhm_right),
        std_hwhm_left_peak_mhz=std_mhz_or_none(hwhm_left),
        std_hwhm_right_peak_mhz=std_mhz_or_none(hwhm_right),

        cov_freq_left_right=cov_poly,
        corr_freq_left_right=corr_poly,

        cov_freq_left_right_interp=cov_interp,
        corr_freq_left_right_interp=corr_interp,
    )