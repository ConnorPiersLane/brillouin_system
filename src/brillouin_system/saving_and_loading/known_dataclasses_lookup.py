from brillouin_system.my_dataclasses.measurements import (
    MeasurementSeries, MeasurementPoint, MeasurementSettings
)
from brillouin_system.my_dataclasses.calibration import (
    CalibrationData, MeasurementsPerFreq, CalibrationMeasurementPoint, CalibrationPolyfitParameters
)
from brillouin_system.my_dataclasses.state_mode import StateMode
from brillouin_system.my_dataclasses.camera_settings import AndorCameraSettings
from brillouin_system.my_dataclasses.background_image import ImageStatistics, BackgroundImage
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum, DisplayResults

known_classes = {
    cls.__name__: cls
    for cls in [
        MeasurementSeries, MeasurementPoint, MeasurementSettings,
        CalibrationData, MeasurementsPerFreq, CalibrationMeasurementPoint,
        CalibrationPolyfitParameters,
        StateMode, AndorCameraSettings,
        ImageStatistics, BackgroundImage,
        ZaberPosition,
        FittedSpectrum, DisplayResults
    ]
}
