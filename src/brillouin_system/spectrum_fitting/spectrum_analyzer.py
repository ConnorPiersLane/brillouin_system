import math
from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts
from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.analyze_util import get_background_values
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts

@dataclass
class TheoreticalPeakStdError:
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
    x1_std: float
    x2_std: float
    D_std: float
    C_std: float
    cov_x1x2: float
    freq_shift_left_peak_ghz_std: float
    freq_shift_right_peak_ghz_std: float
    freq_shift_peak_distance_ghz_std: float
    freq_shift_dc_ghz_std: float
    freq_shift_centroid_ghz_std: float


class SpectrumAnalyzer:
    def __init__(self, calibration_calculator: CalibrationCalculator):
        self.calibration_calculator = calibration_calculator


    def analyze_spectrum(self, fitting: FittedSpectrum) -> AnalyzedFreqShifts:
        if not fitting.is_success:
            return AnalyzedFreqShifts(
                freq_shift_left_peak_ghz=None,
                freq_shift_right_peak_ghz=None,
                freq_shift_peak_distance_ghz=None,
                fwhm_left_peak_ghz=None,
                fwhm_right_peak_ghz=None,
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
            fwhm_left_peak_ghz=float(
                abs(
                    self.calibration_calculator.df_left_peak(
                        fitting.left_peak_center_px,
                        fitting.left_peak_width_px
                    )
                )
            ),
            fwhm_right_peak_ghz=float(
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
            freq_shift_centroid_ghz=float(self.calibration_calculator.freq_peak_centroid((fitting.right_peak_center_px+fitting.left_peak_center_px)/2))

        )


    def theoretical_precision(self, fs: FittedSpectrum, photons: PhotonsCounts, bg_frame):
        """ See paper: Precise nanometer Localization Analysis for Individual Fluorescent Probes """

        analyzed_spec = self.analyze_spectrum(fitting=fs)

        # All values are in GHz, as this is a distance approx for the spectrometer
        # Lorentzian Profile, approximate std with fwhm
        s_l, s_r = analyzed_spec.fwhm_left_peak_ghz, analyzed_spec.fwhm_right_peak_ghz

        a_l = self.calibration_calculator.df_left_peak(px=fs.left_peak_center_px, dpx=1)
        a_r = self.calibration_calculator.df_left_peak(px=fs.right_peak_center_px, dpx=1)

        N_l = photons.left_peak_photons
        N_r = photons.right_peak_photons

        result = get_background_values(bg_frame=bg_frame, fit=fs)
        b_l, b_r = result['left_peak_bg'], result['right_peak_bg']

        dx_l_photons = math.sqrt(            s_l ** 2 / N_l        )
        dx_l_pixelation = math.sqrt(  (a_l**2/12) / N_l)
        dx_l_bg =math.sqrt(  4*math.sqrt(math.pi) * s_l**3*b_l**2 / (a_l * N_l**2) )
        dx_l_total = dx_l_photons + dx_l_pixelation + dx_l_bg

        dx_r_photons = math.sqrt(            s_r ** 2 / N_r        )
        dx_r_pixelation = math.sqrt(  (a_r**2/12) / N_r)
        dx_r_bg =math.sqrt(  4*math.sqrt(math.pi) * s_r**3*b_r**2 / (a_r * N_r**2) )
        dx_r_total = dx_r_photons + dx_r_pixelation + dx_r_bg

        return TheoreticalPeakStdError(
            left_peak_photons= dx_l_photons,
            left_peak_pixelation= dx_l_pixelation,
            left_peak_bg= dx_l_bg,
            left_peak_total= dx_l_total,
            right_peak_photons= dx_r_photons,
            right_peak_pixelation= dx_r_pixelation,
            right_peak_bg= dx_r_bg,
            right_peak_total= dx_r_total,
        )

    def measured_precision(self, fs: list[FittedSpectrum],
                           shifts: list[AnalyzedFreqShifts]) -> MeasuredStatistics | None:
        if not fs or not shifts:
            return None

        x1 = [f.left_peak_center_px for f in fs]
        x2 = [f.right_peak_center_px for f in fs]
        D = [f.inter_peak_distance for f in fs]
        C = [(f.left_peak_center_px + f.right_peak_center_px) / 2 for f in fs]

        # Covariance matrix (2x2)
        cov_matrix = np.cov(x1, x2, bias=False)

        # Extract covariance and variances
        cov_x1x2 = cov_matrix[0, 1]
        x1_std = math.sqrt(cov_matrix[0, 0])
        x2_std = math.sqrt(cov_matrix[1, 1])

        # Std for derived quantities
        D_std_m = float(np.std(D, ddof=1))  # sample std
        C_std_m = float(np.std(C, ddof=1))

        # Std for frequency shifts
        freq_shift_left_peak_ghz_std = float(
            np.std([s.freq_shift_left_peak_ghz for s in shifts if s.freq_shift_left_peak_ghz is not None], ddof=1))
        freq_shift_right_peak_ghz_std = float(
            np.std([s.freq_shift_right_peak_ghz for s in shifts if s.freq_shift_right_peak_ghz is not None], ddof=1))
        freq_shift_peak_distance_ghz_std = float(
            np.std([s.freq_shift_peak_distance_ghz for s in shifts if s.freq_shift_peak_distance_ghz is not None],
                   ddof=1))
        freq_shift_dc_ghz_std = float(
            np.std([s.freq_shift_dc_ghz for s in shifts if s.freq_shift_dc_ghz is not None], ddof=1))
        freq_shift_centroid_ghz_std = float(
            np.std([s.freq_shift_centroid_ghz for s in shifts if s.freq_shift_centroid_ghz is not None], ddof=1))

        return MeasuredStatistics(
            x1_std=x1_std,
            x2_std=x2_std,
            D_std=D_std_m,
            C_std=C_std_m,
            cov_x1x2=float(cov_x1x2),
            freq_shift_left_peak_ghz_std=freq_shift_left_peak_ghz_std,
            freq_shift_right_peak_ghz_std=freq_shift_right_peak_ghz_std,
            freq_shift_peak_distance_ghz_std=freq_shift_peak_distance_ghz_std,
            freq_shift_dc_ghz_std=freq_shift_dc_ghz_std,
            freq_shift_centroid_ghz_std=freq_shift_centroid_ghz_std,
        )

    # --- New helpers ---
    @staticmethod
    def print_theoretical_precision(tpse: TheoreticalPeakStdError):
        print("==== Theoretical Peak Std Error ====")
        print(f"Left Peak: photons={tpse.left_peak_photons:.4g}, "
              f"pixelation={tpse.left_peak_pixelation:.4g}, "
              f"bg={tpse.left_peak_bg:.4g}, "
              f"total={tpse.left_peak_total:.4g}")
        print(f"Right Peak: photons={tpse.right_peak_photons:.4g}, "
              f"pixelation={tpse.right_peak_pixelation:.4g}, "
              f"bg={tpse.right_peak_bg:.4g}, "
              f"total={tpse.right_peak_total:.4g}")
        print("===================================")

    @staticmethod
    def print_measured_precision(ms: MeasuredStatistics):
        print("==== Measured Statistics ====")
        print(f"x1 std = {ms.x1_std:.4g}, x2 std = {ms.x2_std:.4g}, cov(x1,x2)={ms.cov_x1x2:.4g}")
        print(f"D std = {ms.D_std:.4g}, C std = {ms.C_std:.4g}")
        print(f"Freq shifts: left={ms.freq_shift_left_peak_ghz_std:.4g}, "
              f"right={ms.freq_shift_right_peak_ghz_std:.4g}, "
              f"distance={ms.freq_shift_peak_distance_ghz_std:.4g}, "
              f"dc={ms.freq_shift_dc_ghz_std:.4g}, "
              f"centroid={ms.freq_shift_centroid_ghz_std:.4g}")
        print("===============================")