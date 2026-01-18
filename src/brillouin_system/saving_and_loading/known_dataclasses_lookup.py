from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorCameraInfo
from brillouin_system.eye_tracker.eye_tracker import EyeTrackerRawData
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import Intrinsics, StereoExtrinsics, \
    StereoCalibration
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AxialScan, MeasurementPoint
)
from brillouin_system.calibration.calibration import (
    CalibrationData, MeasurementsPerFreq, CalibrationMeasurementPoint, CalibrationPolyfitParameters
)
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.my_dataclasses.background_image import ImageStatistics, BackgroundImage
from brillouin_system.devices.zaber_engines.zaber_position import ZaberPosition
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

known_classes = {
    cls.__name__: cls
    for cls in [
        AxialScan, MeasurementPoint,
        CalibrationData, MeasurementsPerFreq, CalibrationMeasurementPoint,
        CalibrationPolyfitParameters,
        SystemState,
        ImageStatistics, BackgroundImage,
        ZaberPosition,
        FittedSpectrum, DisplayResults,
        AndorCameraInfo,
        Intrinsics, StereoExtrinsics, StereoCalibration,
        EyeTrackerConfig, EyeTrackerRawData, EyeTrackerResults
    ]
}
