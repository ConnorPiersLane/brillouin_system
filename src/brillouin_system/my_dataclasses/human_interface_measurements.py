"""LEGACY UNPICKLING PATH ONLY — the classes moved to their own modules
2026-08-21 (one dataclass per file):

    AxialScan        -> my_dataclasses.axial_scan
    MeasurementPoint -> my_dataclasses.measurement_point
    SweepCycle       -> my_dataclasses.sweep_cycle
    AnalyzedSpectrum -> analysis.analyzed_spectrum

and the functions moved out:

    calibration_for_scan -> calibration.calibration.calibration_calculator_for_scan
                            (takes the calibration info, not the scan)
    fit_axial_scan       -> analysis.fit_axial_scan
    fitter_for_scan      -> removed; construct SpectrumFitter() directly

Old pickles store classes under THIS module path, so the names below must
keep resolving here. Import from the new modules in all code.
"""
from brillouin_system.analysis.analyzed_spectrum import AnalyzedSpectrum
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.measurement_point import MeasurementPoint
from brillouin_system.my_dataclasses.sweep_cycle import SweepCycle
