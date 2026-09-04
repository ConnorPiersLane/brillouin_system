from dataclasses import dataclass

from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult


@dataclass
class SweepCycle:
    """One in-out cycle of a sweep scan.

    reflection_in / reflection_out are the raw finder results of the inward and
    outward crossing (biases NOT corrected — keep corrections in analysis; the
    per-direction bias is not a settled constant, see 2026-07-30 alternate-mode
    characterization). measurement_index points into AxialScan.measurements for
    the frame taken during this cycle, or is None if the cycle took no frame
    (missed in-crossing). A found=False / gated-out crossing is stored as-is so
    single-crossing fallback cycles stay identifiable in the saved data.
    """
    cycle_index: int
    reflection_in: ReflectionResult | None = None
    reflection_out: ReflectionResult | None = None
    measurement_index: int | None = None
