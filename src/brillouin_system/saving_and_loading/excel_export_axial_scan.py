from dataclasses import dataclass
from typing import Optional

import numpy as np




@dataclass
class BrillouinExport:
    axial_scan_i: int
    id: str
    measurement_index: int
    x_mm: Optional[float]
    y_mm: Optional[float]
    z_mm: Optional[float]
    pf_um: Optional[float]
    pb_um: Optional[float]
    m_um: Optional[float]
    lp_ghz: Optional[float]
    lp_hwhm_ghz: Optional[float]
    lp_photons: Optional[float]
    lp_theo_std_mhz: Optional[float]
    rp_ghz: Optional[float]
    rp_hwhm_ghz: Optional[float]
    rp_photons: Optional[float]
    rp_theo_std_mhz: Optional[float]
    distance_ghz: Optional[float]

#
# @dataclass
# class TimeingExports:



