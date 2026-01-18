import math
from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts
from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.analyze_util import get_b_values
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, count_to_electrons


@dataclass
class TheoreticalPeakStdError:
    """ All Values in MHz"""
    left_peak_photons: float
    left_peak_pixelation: float
    left_peak_bg: float
    left_peak_total: float
    right_peak_photons: float
    right_peak_pixelation: float
    right_peak_bg: float
    right_peak_total: float



@dataclass
class MeasuredStatistics:
    """ All Values in MHz, MHz², or -"""
    freq_shift_left_peak_mean_ghz: float
    freq_shift_right_peak_mean_ghz: float
    freq_shift_peak_distance_mean_ghz: float
    freq_shift_dc_mean_ghz: float
    freq_shift_centroid_mean_ghz: float
    freq_shift_left_peak_mhz_std: float
    freq_shift_right_peak_mhz_std: float
    freq_shift_peak_distance_mhz_std: float
    freq_shift_dc_mhz_std: float
    freq_shift_centroid_mhz_std: float
    freq_cov_left_right: float  # MHz²
    freq_corr_left_right: float # Dimensionless


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
                freq_shift_dc_ghz=None,
                freq_shift_centroid_ghz=None,
            )

        return AnalyzedFreqShifts(
            freq_shift_left_peak_ghz=float(
                self.calibration_calculator.freq_left_peak(fitting.left_peak_center_px)
            ),
            freq_shift_right_peak_ghz=float(
                self.calibration_calculator.freq_right_peak(fitting.right_peak_center_px)
            ),
            freq_shift_peak_distance_ghz=float(
                self.calibration_calculator.freq_peak_distance(fitting.inter_peak_distance)
            ),
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
            freq_shift_dc_ghz=self.calibration_calculator.freq_DC_model(
                D=fitting.inter_peak_distance,
                C=(fitting.right_peak_center_px+fitting.left_peak_center_px)/2,
            ),
            freq_shift_centroid_ghz=self.calibration_calculator.freq_peak_centroid((fitting.right_peak_center_px+fitting.left_peak_center_px)/2)

        )


    def theoretical_precision(self, fs: FittedSpectrum,
                              photons: PhotonsCounts,
                              bg_frame_std,
                              preamp_gain: int | float,
                              emccd_gain: int | float,
                              ):
        """ See paper: Precise nanometer Localization Analysis for Individual Fluorescent Probes """

        analyzed_spec = self.analyze_spectrum(fitting=fs)

        # All values are in GHz, as this is a distance approx for the spectrometer
        # Lorentzian Profile, approximate std with fwhm
        s_l, s_r = analyzed_spec.hwhm_left_peak_ghz, analyzed_spec.hwhm_right_peak_ghz

        a_l = self.calibration_calculator.df_left_peak(px=fs.left_peak_center_px, dpx=1)
        a_r = self.calibration_calculator.df_left_peak(px=fs.right_peak_center_px, dpx=1)

        N_l = photons.left_peak_photons
        N_r = photons.right_peak_photons

        if bg_frame_std is None:
            b_counts_l, b_counts_r = 0, 0
        else:
            b_counts_l, b_counts_r = get_b_values(std_image=bg_frame_std, fit=fs)
        b_l = count_to_electrons(b_counts_l, preamp_gain=preamp_gain, emccd_gain=emccd_gain)
        b_r = count_to_electrons(b_counts_r, preamp_gain=preamp_gain, emccd_gain=emccd_gain)

        dx_l_photons = math.sqrt(            s_l ** 2 / N_l        )
        dx_l_pixelation = math.sqrt(  (a_l**2/12) / N_l)
        dx_l_bg =math.sqrt(  4*math.sqrt(math.pi) * s_l**3*b_l**2 / (a_l * N_l**2) )
        dx_l_total =math.sqrt( dx_l_photons**2 + dx_l_pixelation**2 + dx_l_bg**2 )

        dx_r_photons = math.sqrt(            s_r ** 2 / N_r        )
        dx_r_pixelation = math.sqrt(  (a_r**2/12) / N_r)
        dx_r_bg =math.sqrt(  4*math.sqrt(math.pi) * s_r**3*b_r**2 / (a_r * N_r**2) )
        dx_r_total =math.sqrt( dx_r_photons**2 + dx_r_pixelation**2 + dx_r_bg**2)

        return TheoreticalPeakStdError(
            left_peak_photons= dx_l_photons*1000,
            left_peak_pixelation= dx_l_pixelation*1000,
            left_peak_bg= dx_l_bg*1000,
            left_peak_total= dx_l_total*1000,
            right_peak_photons= dx_r_photons*1000,
            right_peak_pixelation= dx_r_pixelation*1000,
            right_peak_bg= dx_r_bg*1000,
            right_peak_total= dx_r_total*1000,
        )


    def measured_precision(self, shifts: list[AnalyzedFreqShifts]) -> MeasuredStatistics | None:
        if not shifts:
            return None

        # Collect arrays (GHz values)
        left_shifts = [s.freq_shift_left_peak_ghz for s in shifts if s.freq_shift_left_peak_ghz is not None]
        right_shifts = [s.freq_shift_right_peak_ghz for s in shifts if s.freq_shift_right_peak_ghz is not None]
        dist_shifts = [s.freq_shift_peak_distance_ghz for s in shifts if s.freq_shift_peak_distance_ghz is not None]
        dc_shifts = [s.freq_shift_dc_ghz for s in shifts if s.freq_shift_dc_ghz is not None]
        centroid_shifts = [s.freq_shift_centroid_ghz for s in shifts if s.freq_shift_centroid_ghz is not None]

        # Mean values
        freq_shift_left_peak_mean_ghz = float(np.mean(left_shifts)) if len(left_shifts) > 1 else None
        freq_shift_right_peak_mean_ghz = float(np.mean(right_shifts)) if len(right_shifts) > 1 else None
        freq_shift_peak_distance_mean_ghz = float(np.mean(dist_shifts)) if len(dist_shifts) > 1 else None
        freq_shift_dc_mean_ghz = float(np.mean(dc_shifts)) if len(dc_shifts) > 1 else None
        freq_shift_centroid_mean_ghz = float(np.mean(centroid_shifts)) if len(centroid_shifts) > 1 else None

        # --- Std devs in MHz ---
        freq_shift_left_peak_std_mhz = float(np.std(left_shifts, ddof=1) * 1e3) if len(left_shifts) > 1 else None
        freq_shift_right_peak_std_mhz = float(np.std(right_shifts, ddof=1) * 1e3) if len(right_shifts) > 1 else None
        freq_shift_peak_distance_std_mhz = float(np.std(dist_shifts, ddof=1) * 1e3) if len(dist_shifts) > 1 else None
        freq_shift_dc_std_mhz = float(np.std(dc_shifts, ddof=1) * 1e3) if len(dc_shifts) > 1 else None
        freq_shift_centroid_std_mhz = float(np.std(centroid_shifts, ddof=1) * 1e3) if len(centroid_shifts) > 1 else None

        # --- Covariance & correlation (MHz² and dimensionless) ---
        freq_cov_left_right_mhz2 = None
        freq_corr_left_right = None
        if len(left_shifts) > 1 and len(right_shifts) > 1:
            cov_matrix = np.cov(left_shifts, right_shifts, ddof=1)
            freq_cov_left_right_mhz2 = float(cov_matrix[0, 1] * 1e6)  # GHz² → MHz²
            if freq_shift_left_peak_std_mhz and freq_shift_right_peak_std_mhz:
                freq_corr_left_right = freq_cov_left_right_mhz2 / (
                        freq_shift_left_peak_std_mhz * freq_shift_right_peak_std_mhz
                )

        return MeasuredStatistics(
            freq_shift_left_peak_mean_ghz = freq_shift_left_peak_mean_ghz,
            freq_shift_right_peak_mean_ghz = freq_shift_right_peak_mean_ghz,
            freq_shift_peak_distance_mean_ghz = freq_shift_peak_distance_mean_ghz,
            freq_shift_dc_mean_ghz = freq_shift_dc_mean_ghz,
            freq_shift_centroid_mean_ghz = freq_shift_centroid_mean_ghz,
            freq_shift_left_peak_mhz_std=freq_shift_left_peak_std_mhz,
            freq_shift_right_peak_mhz_std=freq_shift_right_peak_std_mhz,
            freq_shift_peak_distance_mhz_std=freq_shift_peak_distance_std_mhz,
            freq_shift_dc_mhz_std=freq_shift_dc_std_mhz,
            freq_shift_centroid_mhz_std=freq_shift_centroid_std_mhz,
            freq_cov_left_right=freq_cov_left_right_mhz2,  # MHz²
            freq_corr_left_right=freq_corr_left_right,  # dimensionless
        )
