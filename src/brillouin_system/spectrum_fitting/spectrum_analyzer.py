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
    distance_total_mhz: float

@dataclass
class AnalyzedFreqShifts:
    freq_shift_left_peak_ghz_poly: float | None
    freq_shift_right_peak_ghz_poly: float | None
    freq_shift_peak_distance_ghz_poly: float | None
    hwhm_left_peak_ghz: float | None
    hwhm_right_peak_ghz: float | None
    freq_shift_left_peak_ghz_interp: float | None = None
    freq_shift_right_peak_ghz_interp: float | None = None
    freq_shift_peak_distance_ghz_interp: float | None = None



class SpectrumAnalyzer:
    def __init__(self, calibration_calculator: CalibrationCalculator):
        self.calibration_calculator = calibration_calculator

    def analyze_spectrum(self, fitting: FittedSpectrum) -> AnalyzedFreqShifts:
        if not fitting.is_success:
            return AnalyzedFreqShifts(
                freq_shift_left_peak_ghz_poly=None,
                freq_shift_right_peak_ghz_poly=None,
                freq_shift_peak_distance_ghz_poly=None,
                hwhm_left_peak_ghz=None,
                hwhm_right_peak_ghz=None,
            )

        return AnalyzedFreqShifts(
            freq_shift_left_peak_ghz_poly=self.calibration_calculator.freq_left_peak(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz_poly=self.calibration_calculator.freq_right_peak(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz_poly=self.calibration_calculator.freq_peak_distance(fitting.inter_peak_distance),
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

        distance_total = 0.5*math.sqrt(dx_l_total**2+dx_r_total**2)

        return TheoreticalPeakStdError(
            left_peak_photons_mhz=dx_l_photons * 1000,
            left_peak_pixelation_mhz=dx_l_pixelation * 1000,
            left_peak_bg_mhz=dx_l_bg * 1000,
            left_peak_total_mhz=dx_l_total * 1000,
            right_peak_photons_mhz=dx_r_photons * 1000,
            right_peak_pixelation_mhz=dx_r_pixelation * 1000,
            right_peak_bg_mhz=dx_r_bg * 1000,
            right_peak_total_mhz=dx_r_total * 1000,
            distance_total_mhz=distance_total * 1000,
        )


