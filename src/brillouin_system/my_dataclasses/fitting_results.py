from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


@dataclass
class FittingResults:
    frame: np.ndarray
    fitted_spectrum: FittedSpectrum
    sd: float
    fsr: float
    freq_shift_ghz: float | None
