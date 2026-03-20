from dataclasses import dataclass, fields
from typing import Optional

from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedSpectrum
from dataclasses import asdict
import pandas as pd

@dataclass
class BrillouinExport:
    axial_scan_i: int
    id: str
    measurement_index: int
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None
    z_mm: Optional[float] = None
    pf_um: Optional[float] = None
    pb_um: Optional[float] = None
    m_um: Optional[float] = None
    lp_ghz_poly: Optional[float] = None
    lp_ghz_interp: Optional[float] = None
    lp_hwhm_ghz: Optional[float] = None
    lp_photons: Optional[float] = None
    lp_theo_std_photons_mhz: Optional[float] = None
    lp_theo_std_pixelation_mhz: Optional[float] = None
    lp_theo_std_bg_mhz: Optional[float] = None
    lp_theo_std_total_mhz: Optional[float] = None
    rp_ghz_poly: Optional[float] = None
    rp_ghz_interp: Optional[float] = None
    rp_hwhm_ghz: Optional[float] = None
    rp_photons: Optional[float] = None
    rp_theo_std_photons_mhz: Optional[float] = None
    rp_theo_std_pixelation_mhz: Optional[float] = None
    rp_theo_std_bg_mhz: Optional[float] = None
    rp_theo_std_total_mhz: Optional[float] = None
    distance_ghz_poly: Optional[float] = None
    distance_ghz_interp: Optional[float] = None
    distance_theo_std_mhz: Optional[float] = None
    ts_frame: Optional[float] = None
    ts_pf: Optional[float] = None
    ts_pb: Optional[float] = None


def get_excel_row_data(axial_scan: AxialScan, analyzed_spectrum: AnalyzedSpectrum, idx: int) -> BrillouinExport:

    shifts = analyzed_spectrum.analyzed_shifts
    photons = analyzed_spectrum.photons
    theo = analyzed_spectrum.theoretical_precisions

    # If not None get laserposition
    x_mm = None
    y_mm = None
    z_mm = None
    if axial_scan.eye_tracker_results is not None:
        lp = axial_scan.eye_tracker_results.laser_position
        if axial_scan.eye_tracker_results.laser_position is not None:
            x_mm = lp[0]
            y_mm = lp[1]
            z_mm = lp[2]

    pf_um = None
    ts_pf = None
    if axial_scan.reflection_result_forwards is not None:
        ts_pf = axial_scan.reflection_result_forwards.event_time_perf
        pf_um = axial_scan.reflection_result_forwards.event_z_um

    pb_um = None
    ts_pb = None
    if axial_scan.reflection_result_backwards is not None:
        ts_pb = axial_scan.reflection_result_backwards.event_time_perf
        pb_um = axial_scan.reflection_result_backwards.event_z_um

    measurement_um = axial_scan.measurements[idx].lens_zaber_position



    return BrillouinExport(
        axial_scan_i = axial_scan.i,
        id = axial_scan.id,
        measurement_index = idx,
        x_mm = x_mm,
        y_mm = y_mm,
        z_mm = z_mm,
        pf_um = pf_um,
        pb_um = pb_um,
        m_um = measurement_um,
        # Peak frequencies
        lp_ghz_poly=shifts.freq_shift_left_peak_ghz_poly,
        lp_ghz_interp=shifts.freq_shift_left_peak_ghz_interp,
        rp_ghz_poly=shifts.freq_shift_right_peak_ghz_poly,
        rp_ghz_interp=shifts.freq_shift_right_peak_ghz_interp,

        # Widths
        lp_hwhm_ghz=shifts.hwhm_left_peak_ghz,
        rp_hwhm_ghz=shifts.hwhm_right_peak_ghz,

        # Photon counts
        lp_photons=photons.left_peak_photons,
        rp_photons=photons.right_peak_photons,

        # Theoretical precision
        lp_theo_std_photons_mhz=theo.left_peak_photons_mhz,
        lp_theo_std_pixelation_mhz=theo.left_peak_pixelation_mhz,
        lp_theo_std_bg_mhz=theo.left_peak_bg_mhz,
        lp_theo_std_total_mhz=theo.left_peak_total_mhz,

        rp_theo_std_photons_mhz=theo.right_peak_photons_mhz,
        rp_theo_std_pixelation_mhz=theo.right_peak_pixelation_mhz,
        rp_theo_std_bg_mhz=theo.right_peak_bg_mhz,
        rp_theo_std_total_mhz=theo.right_peak_total_mhz,


        # Distance between peaks
        distance_ghz_poly=shifts.freq_shift_peak_distance_ghz_poly,
        distance_ghz_interp=shifts.freq_shift_peak_distance_ghz_interp,
        distance_theo_std_mhz=theo.distance_total_mhz,
        ts_frame = axial_scan.measurements[idx].time_stamp,
        ts_pf = ts_pf,
        ts_pb = ts_pb,
    )



# assuming your dataclass is already defined
# and you have a list like:
# data: List[BrillouinExport]

def export_to_excel(data: list[BrillouinExport], output_path: str):
    # Convert each dataclass instance to a dictionary
    dict_list = [asdict(item) for item in data]

    # Create DataFrame
    df = pd.DataFrame(dict_list)

    # Export to Excel
    df.to_excel(output_path, index=False)

    print(f"Exported {len(df)} rows to {output_path}")


def load_from_excel(path: str, sheet_name: str | int = 0) -> list[BrillouinExport]:
    df = pd.read_excel(path, sheet_name=sheet_name)

    valid_fields = {f.name for f in fields(BrillouinExport)}

    # Ensure all required columns exist
    for field in valid_fields:
        if field not in df.columns:
            df[field] = None

    df = df[list(valid_fields)]
    df = df.where(pd.notnull(df), None)

    return [
        BrillouinExport(**row.to_dict())
        for _, row in df.iterrows()
    ]

# Example usage:
if __name__ == "__main__":
    data = [
        BrillouinExport(axial_scan_i=1, id="A", measurement_index=0, x_mm=1.2),
        BrillouinExport(axial_scan_i=2, id="B", measurement_index=1, y_mm=3.4),
    ]

    export_to_excel(data, "brillouin_export.xlsx")
    d = load_from_excel("brillouin_export.xlsx")
    pass