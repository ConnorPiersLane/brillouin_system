from dataclasses import dataclass

import numpy as np

from brillouin_system.eye_tracker.eye_position.coordinates import RigPupilTransform, RigCoord, PupilCoord
from brillouin_system.eye_tracker.eye_position.cornea_position import calc_distance_laser_corner


@dataclass
class EyeTrackerResults:
    """
    laser_position [mm]: with respect to the pupil center: xy within the plane, z in gaze direction (outwards, into laser)
    delta_laser_corner [mm]: >0-laser in front of cornea, <0-laser within cornea / eye
    """
    left_img: np.ndarray
    right_img: np.ndarray
    time_stamp: float
    laser_position: None | tuple[float, float, float] #xyz
    delta_laser_corner: None | float

def get_eye_tracker_results(left: np.ndarray,
                            right: np.ndarray,
                            meta: dict,
                            laser_focus_position: RigCoord) -> EyeTrackerResults:
    """
    (left, right, meta)
      left     : np.ndarray (H, W, 3), uint8
      right    : np.ndarray (H, W, 3), uint8
      meta     : Dict: {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}
    """
    ts = meta["ts"]
    pupil3D = meta["pupil3D"]

    if pupil3D is not None:
        transform = RigPupilTransform(pupil_center=RigCoord(x=pupil3D.center_ref[0],
                                                            y=pupil3D.center_ref[1],
                                                            z=pupil3D.center_ref[2])
                                      )
        laser_pupil_coord: PupilCoord = transform.rig_to_pupil(laser_focus_position)
        laser_focus_position = (laser_pupil_coord.x, laser_pupil_coord.y, laser_pupil_coord.z)
        delta_laser_corner = calc_distance_laser_corner(laser_pupil_coord)
    else:
        laser_focus_position = None
        delta_laser_corner = None

    return EyeTrackerResults(
        left_img=left,
        right_img=right,
        time_stamp=ts,
        laser_position=laser_focus_position,
        delta_laser_corner=delta_laser_corner,
    )