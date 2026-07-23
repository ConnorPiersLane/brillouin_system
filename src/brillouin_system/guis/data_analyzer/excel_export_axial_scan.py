from dataclasses import dataclass, fields
from typing import Optional

from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedSpectrum
from dataclasses import asdict
import pandas as pd

@dataclass
class BrillouinExport:
    """
    lp = left peak
    rp = right peak
    distance = distance btw. peaks
    theo = theoretical value
    bg = background
    ts = timestamp
    pf = found reflection plane going forwards (into the cornea) pf=plane forward
    bg = plane backward
    """

    axial_scan_i: int   # Axial scan id: start=0
    id: str  # User specified id
    measurement_index: int  # Index of the measurement taken for that scan. start=0 index = 3 -> fourth measurement
    x_mm: Optional[float] = None # Position in Pupil coordinates. x-> to the right (persons right eye, towards the nose)
    y_mm: Optional[float] = None #upwards
    z_mm: Optional[float] = None # from pupil plane towards laser (looking into the laser)
    pf_um: Optional[float] = None # position of zaber lens when plane was found moving forwards pf=plane forwards
    pb_um: Optional[float] = None # position plane backwards
    m_um: Optional[float] = None # position where the measurement was taken.
    # Ideal (m_um-pf_um) = (m_um-pb_um) = scan depth

    reflection_signal_pf_V: Optional[float] = None # measured reflection signal of the photodiode / daq
    reflection_signal_pb_V: Optional[float] = None

    reflection_signal_bg_mean_V: Optional[float] = None # background of the signal used for pf and pb
    reflection_signal_bg_std_V: Optional[float] = None


    lp_ghz: Optional[float] = None
    # lp_ghz_interp: Optional[float] = None
    lp_hwhm_ghz: Optional[float] = None
    lp_photons: Optional[float] = None
    lp_theo_std_photons_mhz: Optional[float] = None
    lp_theo_std_pixelation_mhz: Optional[float] = None
    lp_theo_std_bg_mhz: Optional[float] = None
    lp_theo_std_total_mhz: Optional[float] = None
    rp_ghz: Optional[float] = None
    # rp_ghz_interp: Optional[float] = None
    rp_hwhm_ghz: Optional[float] = None
    rp_photons: Optional[float] = None
    rp_theo_std_photons_mhz: Optional[float] = None
    rp_theo_std_pixelation_mhz: Optional[float] = None
    rp_theo_std_bg_mhz: Optional[float] = None
    rp_theo_std_total_mhz: Optional[float] = None
    distance_ghz: Optional[float] = None
    # distance_ghz_interp: Optional[float] = None
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
    reflection_signal_pf_v = None
    pf_background_mean = None
    pf_background_std = None
    reflection_signal_pf_v = None
    if axial_scan.reflection_result_forwards is not None:
        ts_pf = axial_scan.reflection_result_forwards.event_time_perf
        pf_um = axial_scan.reflection_result_forwards.event_z_um
        reflection_signal_pf_v = axial_scan.reflection_result_forwards.peak_value
        pf_background_mean = axial_scan.reflection_result_forwards.background_mean
        pf_background_std = axial_scan.reflection_result_forwards.background_std

    pb_um = None
    ts_pb = None
    reflection_signal_pb_v = None
    if axial_scan.reflection_result_backwards is not None:
        ts_pb = axial_scan.reflection_result_backwards.event_time_perf
        pb_um = axial_scan.reflection_result_backwards.event_z_um
        reflection_signal_pb_v = axial_scan.reflection_result_backwards.peak_value

    measurement_um = axial_scan.measurements[idx].lens_zaber_position



    return BrillouinExport(
        axial_scan_i = axial_scan.i,
        id = axial_scan.id,
        measurement_index = idx,

        # Eye Position
        x_mm = x_mm,
        y_mm = y_mm,
        z_mm = z_mm,

        # Reflection Plane finding forwards
        pf_um = pf_um,
        reflection_signal_pf_V = reflection_signal_pf_v,

        # Reflection Plane finding backwards
        pb_um = pb_um,
        reflection_signal_pb_V=reflection_signal_pb_v,

        reflection_signal_bg_mean_V=pf_background_mean,
        reflection_signal_bg_std_V=pf_background_std,

        # Measurement axial position
        m_um=measurement_um,

        # Peak frequencies
        lp_ghz=shifts.freq_shift_left_peak_ghz,
        # lp_ghz_interp=shifts.freq_shift_left_peak_ghz_interp,
        rp_ghz=shifts.freq_shift_right_peak_ghz,
        # rp_ghz_interp=shifts.freq_shift_right_peak_ghz_interp,

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
        distance_ghz=shifts.freq_shift_peak_distance_ghz,
        # distance_ghz_interp=shifts.freq_shift_peak_distance_ghz_interp,
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