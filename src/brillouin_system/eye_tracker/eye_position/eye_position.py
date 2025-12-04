from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord, RigPupilTransform, PupilCoord
from brillouin_system.eye_tracker.eye_position.cornea_position import calc_anterior_cornea_z_position


class EyePosition:

    def __init__(self, pupil_center: RigCoord, laser_focus_position: RigCoord):
        self.t = RigPupilTransform(pupil_center=pupil_center)
        self.laser_position_pupil_coord: PupilCoord = self.t.rig_to_pupil(r = laser_focus_position)

        cornea_height = calc_anterior_cornea_z_position(x=self.laser_position_pupil_coord.x,
                                                        y=self.laser_position_pupil_coord.y)
        self.distance_laser_cornea = self.laser_position_pupil_coord.z - cornea_height

