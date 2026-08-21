from dataclasses import dataclass

from brillouin_system.calibration.calibration import AnalyzedFreqShifts
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.analysis.pixel_counts_and_photons import PixelCountsAndPhotons
from brillouin_system.analysis.thompson_shot_noise_limit import TheoreticalPeakStdError


@dataclass
class AnalyzedSpectrum:
    """Everything one frame's fit yields: the fit itself, its GHz conversion,
    the photon numbers behind it, and the Thompson precision bound."""
    fitted_spectrum: FittedSpectrum
    analyzed_shifts: AnalyzedFreqShifts
    photons: PixelCountsAndPhotons
    theoretical_precisions: TheoreticalPeakStdError
