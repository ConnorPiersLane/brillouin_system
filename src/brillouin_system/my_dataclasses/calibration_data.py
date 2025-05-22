from dataclasses import dataclass

from brillouin_system.my_dataclasses.fitting_results import FittingResults

@dataclass
class CalibrationData:
    reference_freqs_ghz: list[float]
    data: list[FittingResults]
    sd: float
    fsr: float