from brillouin_system.eye_tracker.et2_pupil_fitting.pupil_fitting_config.object_detection_config import \
    left_cam_detect_pupil_config, right_cam_detect_pupil_config, left_cam_detect_glint_config, \
    right_cam_detect_glint_config


class ObjectDetector:

    def __init__(self):
        l_pupil_config = left_cam_detect_pupil_config.get()
        r_pupil_config = right_cam_detect_pupil_config.get()
        l_glint_config = left_cam_detect_glint_config.get()
        r_glint_config = right_cam_detect_glint_config.get()

    def find_ellipse_left_cam(self, ):



