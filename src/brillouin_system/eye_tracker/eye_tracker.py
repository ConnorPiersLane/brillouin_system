from dataclasses import dataclass
from typing import Literal


import numpy as np

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig, eye_tracker_config
from brillouin_system.eye_tracker.eye_tracker_helpers import scale_image, make_black_image, \
    ensure_uint8, draw_ellipse_rgb, gray_to_rgb
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse, PupilImgType
from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
from brillouin_system.eye_tracker.pupil_fitting.pupil_detector import PupilDetector
from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import StereoCalibration
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras
from brillouin_system.eye_tracker.stereo_imaging.init_se3 import left_to_ref
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5
from brillouin_system.eye_tracker.eye_renderer.renderer import EyeSceneRenderer

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


IMG_SIZE = (512, 512)
ELLIPSE_OVERLAY_COLOR = (0, 255, 0)  # RGB: green
ELLIPSE_OVERLAY_THICKNESS = 3


RETURN_FRAME_MAPPING = {
    "original": PupilImgType.ORIGINAL,
    "binary": PupilImgType.BINARY,
    "floodfilled": PupilImgType.FLOODFILLED,
    "contour": PupilImgType.CONTOUR,
}

class EyeTracker:
    def __init__(self, use_dummy=False):
        self._save_data = False
        self._save_data_timestamps: list[float] = []
        self._save_data_N: int = 0
        self._stored_data: list[EyeTrackerRawData] = []


        self.pupil_detector = PupilDetector(stereo_cameras=stereo_cameras, left_to_ref=left_to_ref)
        self.ellipse_fitter = EllipseFitter()

        self.config = None
        self._min_delta_time = 1
        config_init = eye_tracker_config.get()
        self.set_config(config=config_init)

        self.renderer = EyeSceneRenderer(window_size=IMG_SIZE)  # (512, 512)

        self.dual_cam_proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.dual_cam_proxy.start()

    def set_config(self, config: EyeTrackerConfig):
        self.config = config
        self._min_delta_time = 1 / config.max_saving_freq_hz
        self.ellipse_fitter.set_binary_thresholds(
            binary_threshold_left=config.binary_threshold_left,
            binary_threshold_right=config.binary_threshold_right,
        )
        self._set_img_return_typ(frame_returned=config.frame_returned)

    def _set_img_return_typ(self, frame_returned: Literal["original", "binary", "floodfilled", "contour"] = "original"):
        frame_returned = str(frame_returned).strip().lower()
        if frame_returned not in RETURN_FRAME_MAPPING:
            raise ValueError(f"Invalid frame_returned value: {frame_returned}")

        self.ellipse_fitter.set_img_return_type(img_return_type=RETURN_FRAME_MAPPING[frame_returned])

    def _get_settings(self) -> EyeTrackerSettings:
        return EyeTrackerSettings(
            config=self.config,
            stereo_calibration=self.pupil_detector.stereo.st_cal,
        )

    def _clear_stored_data(self):
        self._stored_data.clear()
        self._save_data_timestamps.clear()
        self._save_data_N = 0

    def start_saving(self):
        self._save_data = True

    def end_saving(self):
        self._save_data = False
        self._save_data_to_disk()
        self._clear_stored_data()

    def _get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        left_img, right_img, meta = self.dual_cam_proxy.get_frames()
        return left_img, right_img, float(meta["ts"])  # ts from time.time()


    def _get_ellipses(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple[PupilEllipse, PupilEllipse]:
        pupil_eL: PupilEllipse = self.ellipse_fitter.find_pupil_left(left_img)
        pupil_eR: PupilEllipse = self.ellipse_fitter.find_pupil_right(right_img)
        return pupil_eL, pupil_eR

    def _get_pupil3D(self, eL: Ellipse2D | None, eR: Ellipse2D | None) -> Pupil3D:
        return self.pupil_detector.triangulate_center_using_cones(eL=eL, eR=eR)

    def _get_cam_imgs_for_display(self, pupil_eL: PupilEllipse, pupil_eR: PupilEllipse) -> tuple[np.ndarray, np.ndarray]:
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

        # select source images
        left_img = pupil_eL.pupil_img
        right_img = pupil_eR.pupil_img

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

    def _get_rendered_eye_image(self, pupil3D):
        if pupil3D is None:
            return make_black_image()

        C, gaze = self._pose_from_pupil3D(pupil3D)
        if C is None:
            return make_black_image()

        self.renderer.set_eye_pose(C, gaze)
        return ensure_uint8(self.renderer.get_img())



    # ---------------------------
    # rate-limited saver
    # ---------------------------
    def _should_save(self, timestamp: float) -> bool:
        if not self._save_data:
            return False
        if not self._save_data_timestamps:
            return True
        return (timestamp - self._save_data_timestamps[-1]) >= self._min_delta_time



    def _add_data_for_saving(self, timestamp: float,
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

    def _save_data_to_disk(self) -> None:
        results = EyeTrackerResultsForDiskSave(
            settings=self._get_settings(),
            rawdata=list(self._stored_data),
        )
        native = dataclass_to_hdf5_native_dict(results)
        save_dict_to_hdf5(self.config.save_images_path, native)

    def _pose_from_pupil3D(self, pupil3D: Pupil3D):
        """
        Convert Pupil3D (with .center_ref and .normal_ref) into a pose
        for the renderer.
        """
        if pupil3D is None or pupil3D.center_ref is None:
            return None, None

        C = pupil3D.center_ref.astype(float)

        if pupil3D.normal_ref is not None:
            gaze = pupil3D.normal_ref.astype(float)
            gaze /= np.linalg.norm(gaze) + 1e-9
        else:
            gaze = np.array([0.0, 0.0, 1.0])  # fallback

        # Use renderer’s own geometry to push the cornea center slightly forward
        cornea_forward_mm = (
                self.renderer.parts["z_apex"] - self.renderer.parts["z_iris"]
        )
        C_cornea = C + gaze * cornea_forward_mm

        return C_cornea, gaze

    def get_results_for_gui(self) -> EyeTrackerResultsForGui:
        left_img, right_img, ts = self._get_frames()

        _should_save = self._should_save(timestamp=ts)

        if _should_save:
            left_img_original = left_img.copy()
            right_img_original = right_img.copy()
        else:
            left_img_original = None
            right_img_original = None

        # Ensure they are all rgb
        if self.config.do_ellipse_fitting:
            pupil_eL, pupil_eR = self._get_ellipses(left_img, right_img)
            pupil_3D = self._get_pupil3D(pupil_eL.ellipse, pupil_eR.ellipse)
            cam_left_img, cam_right_img = self._get_cam_imgs_for_display(pupil_eL, pupil_eR)
            rendered_img = self._get_rendered_eye_image(pupil_3D)
        else:
            pupil_3D=None
            cam_left_img = gray_to_rgb(ensure_uint8(left_img))
            cam_right_img = gray_to_rgb(ensure_uint8(right_img))
            rendered_img = make_black_image()

        if _should_save:
            self._add_data_for_saving(timestamp=ts,
                                      left_img_original=left_img_original,
                                      right_img_original=right_img_original,
                                      pupil3D=pupil_3D)


        # Ensure all images are corrctly scaled
        # ensure correct scaling and rgb (for shared memory space) for GUI display
        cam_left_img = scale_image(cam_left_img, IMG_SIZE)
        cam_right_img = scale_image(cam_right_img, IMG_SIZE)
        rendered_img = scale_image(rendered_img, IMG_SIZE)

        #
        cam_left_img = ensure_uint8(cam_left_img)
        cam_right_img = ensure_uint8(cam_right_img)
        rendered_img = ensure_uint8(rendered_img)


        return EyeTrackerResultsForGui(
            pupil3D = pupil_3D,
            cam_left_img=cam_left_img,
            cam_right_img=cam_right_img,
            rendered_img=rendered_img,
        )

    def set_allied_vision_configs(self, cfg_left: AlliedConfig, cfg_right: AlliedConfig):
        self.dual_cam_proxy.set_configs(cfg_left=cfg_left, cfg_right=cfg_right)


    def shutdown(self):
        self.dual_cam_proxy.shutdown()
