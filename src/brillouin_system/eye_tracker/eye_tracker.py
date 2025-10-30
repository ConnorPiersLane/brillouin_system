from dataclasses import dataclass
from typing import Literal


import numpy as np

from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_helpers import scale_image, make_black_image, \
    ensure_uint8, draw_ellipse_rgb, gray_to_rgb
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse
from brillouin_system.eye_tracker.pupil_fitting.pupil_detector import Pupil3D, PupilDetector
from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import StereoCalibration
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras
from brillouin_system.eye_tracker.stereo_imaging.init_se3 import left_to_ref
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5


@dataclass
class EyeTrackerSettings:
    config: EyeTrackerConfig
    stereo_calibration: StereoCalibration


@dataclass
class EyeTrackerRawData:
    timestamp: float
    left_img_original: np.ndarray
    right_img_original: np.ndarray
    pupil3D: Pupil3D | None

@dataclass
class EyeTrackerResultsForDiskSave:
    settings: EyeTrackerSettings
    rawdata: list[EyeTrackerRawData]

@dataclass
class EyeTrackerResultsForGui:
    pupil3D: Pupil3D | None
    cam_left_img: np.ndarray
    cam_right_img: np.ndarray
    rendered_img: np.ndarray
    xymap_img: np.ndarray


IMG_SIZE = (512, 512)
ELLIPSE_OVERLAY_COLOR = (0, 255, 0)  # RGB: green
ELLIPSE_OVERLAY_THICKNESS = 3

class EyeTracker:
    def __init__(self, config: EyeTrackerConfig, use_dummy=False):
        self._save_data = False
        self._save_data_timestamps: list[float] = []
        self._min_delta_time = 1 / config.max_saving_freq_hz  # seconds
        self._save_data_N: int = 0
        self._stored_data: list[EyeTrackerRawData] = []

        self.config = config

        self.pupil_detector = PupilDetector(stereo_cameras=stereo_cameras, left_to_ref=left_to_ref)
        self.ellipse_fitter = EllipseFitter(
            binary_threshold_left=config.binary_threshold_left,
            binary_threshold_right=config.binary_threshold_right,
        )

        self.dual_cam_proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.dual_cam_proxy.start()

    def set_configs(self, config: EyeTrackerConfig):
        self.config = config
        self._min_delta_time = 1 / config.max_saving_freq_hz
        self.ellipse_fitter.set_binary_thresholds(
            binary_threshold_left=config.binary_threshold_left,
            binary_threshold_right=config.binary_threshold_right,
        )

    def get_settings(self) -> EyeTrackerSettings:
        return EyeTrackerSettings(
            config=self.config,
            stereo_calibration=self.pupil_detector.stereo.st_cal,
        )

    def clear_stored_data(self):
        self._stored_data.clear()
        self._save_data_timestamps.clear()
        self._save_data_N = 0

    def start_saving(self):
        self._save_data = True

    def end_saving(self):
        self._save_data = False
        self.save_data_to_disk()
        self.clear_stored_data()

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        left_img, right_img, meta = self.dual_cam_proxy.get_frames()
        return left_img, right_img, float(meta["ts"])  # ts from time.time()


    def get_ellipses(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple[PupilEllipse, PupilEllipse]:
        pupil_eL: PupilEllipse = self.ellipse_fitter.find_pupil_left(left_img)
        pupil_eR: PupilEllipse = self.ellipse_fitter.find_pupil_right(right_img)
        return pupil_eL, pupil_eR

    def get_pupil3D(self, eL: Ellipse2D | None, eR: Ellipse2D | None) -> Pupil3D:
        return self.pupil_detector.triangulate_center_using_cones(eL=eL, eR=eR)

    def get_cam_imgs_for_display(self, pupil_eL, pupil_eR) -> tuple[np.ndarray, np.ndarray]:
        """
        Return display-ready left and right images according to current config.

        Parameters
        ----------
        pupil_eL, pupil_eR : objects from EllipseFitter
            Contain .original_img, .binary_img, .floodfilled_img, etc.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (left_img, right_img) scaled for display.
        """
        frame_to_be_returned: Literal["original", "binary", "floodfilled"] = self.config.frame_returned

        # select source images
        if frame_to_be_returned == "original":
            left_img = pupil_eL.original_img
            right_img = pupil_eR.original_img
        elif frame_to_be_returned == "binary":
            left_img = pupil_eL.binary_img
            right_img = pupil_eR.binary_img
        elif frame_to_be_returned == "floodfilled":
            left_img = pupil_eL.floodfilled_img
            right_img = pupil_eR.floodfilled_img
        else:
            raise ValueError(f"Unknown frame type '{frame_to_be_returned}'")


        # Convert to uint8
        left_img = ensure_uint8(left_img)
        right_img = ensure_uint8(right_img)

        # Ensure RGB
        left_img = gray_to_rgb(left_img)
        right_img = gray_to_rgb(right_img)

        if self.config.overlay_ellipse:
            if pupil_eL.ellipse is not None:
                left_img = draw_ellipse_rgb(img_rgb=left_img,
                                           ellipse=pupil_eL.ellipse,
                                           color_rgb = ELLIPSE_OVERLAY_COLOR,  # green
                                           thickness = ELLIPSE_OVERLAY_THICKNESS, # int
                                           )

            if pupil_eR.ellipse is not None:
                right_img = draw_ellipse_rgb(img_rgb=right_img,
                                           ellipse=pupil_eR.ellipse,
                                           color_rgb = ELLIPSE_OVERLAY_COLOR,  # green
                                           thickness = ELLIPSE_OVERLAY_THICKNESS, # int
                                           )



        return left_img, right_img


    def get_rendered_eye_image(self, pupil3D: Pupil3D) -> np.ndarray:
        # Must return uint8 rgb
        return make_black_image() # note for chatgpt: ok for now, beta version

    def get_xymap_img_image(self, pupil3D: Pupil3D) -> np.ndarray:
        # Must return uint8 rgb
        return make_black_image() # note for chatgpt: ok for now, beta version


    # ---------------------------
    # rate-limited saver
    # ---------------------------
    def _should_save(self, timestamp: float) -> bool:
        if not self._save_data:
            return False
        if not self._save_data_timestamps:
            return True
        return (timestamp - self._save_data_timestamps[-1]) >= self._min_delta_time



    def add_data_for_saving(self, timestamp: float,
                  left_img_original: np.ndarray,
                  right_img_original: np.ndarray,
                  pupil3D: Pupil3D | None):

        self._stored_data.append(
            EyeTrackerRawData(
                timestamp=timestamp,  # now actually seconds, but you can rename field later
                left_img_original=left_img_original,
                right_img_original=right_img_original,
                pupil3D=pupil3D,
            )
        )
        self._save_data_timestamps.append(timestamp)
        self._save_data_N += 1

    def save_data_to_disk(self) -> None:
        results = EyeTrackerResultsForDiskSave(
            settings=self.get_settings(),
            rawdata=list(self._stored_data),
        )
        native = dataclass_to_hdf5_native_dict(results)
        save_dict_to_hdf5(self.config.save_images_path, native)



    def get_display_frames(self):
        left_img, right_img, ts = self.get_frames()

        _should_save = self._should_save(timestamp=ts)

        if _should_save:
            left_img_original = left_img.copy()
            right_img_original = right_img.copy()
        else:
            left_img_original = None
            right_img_original = None

        if self.config.do_ellipse_fitting:
            pupil_eL, pupil_eR = self.get_ellipses(left_img, right_img)
            pupil_3D = self.get_pupil3D(pupil_eL.ellipse, pupil_eR.ellipse)
            cam_left_img, cam_right_img = self.get_cam_imgs_for_display(pupil_eL, pupil_eR)
            rendered_img = self.get_rendered_eye_image(pupil_3D)
            xymap_img = self.get_xymap_img_image(pupil_3D)
        else:
            pupil_3D=None
            cam_left_img = gray_to_rgb(ensure_uint8(left_img))
            cam_right_img = gray_to_rgb(ensure_uint8(right_img))
            rendered_img = make_black_image()
            xymap_img = make_black_image()

        if _should_save:
            self.add_data_for_saving(timestamp=ts,
                                     left_img_original=left_img_original,
                                     right_img_original=right_img_original,
                                     pupil3D=pupil_3D)


        # Ensure all images are corrctly scaled and rgb
        # ensure correct scaling and rgb (for shared memory space) for GUI display
        cam_left_img = scale_image(cam_left_img, IMG_SIZE)
        cam_right_img = scale_image(cam_right_img, IMG_SIZE)
        rendered_img = scale_image(rendered_img, IMG_SIZE)
        xymap_img = scale_image(xymap_img, IMG_SIZE)

        #
        cam_left_img = ensure_uint8(cam_left_img)
        cam_right_img = ensure_uint8(cam_right_img)
        rendered_img = ensure_uint8(rendered_img)
        xymap_img = ensure_uint8(xymap_img)


        return EyeTrackerResultsForGui(
            pupil3D = pupil_3D,
            cam_left_img=cam_left_img,
            cam_right_img=cam_right_img,
            rendered_img=rendered_img,
            xymap_img=xymap_img,
        )
