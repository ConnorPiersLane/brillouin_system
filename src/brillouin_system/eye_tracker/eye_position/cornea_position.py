import math

from brillouin_system.eye_tracker.eye_position.coordinates import PupilCoord


def calc_anterior_cornea_z_position(x, y):
    r = 7.8   # anterior corneal radius (mm)
    acd = 3.0 # anterior chamber depth (mm)

    rr = x*x + y*y
    if rr > r*r:
        return None  # outside spherical surface

    # sphere center is at z = acd - r, pupil plane at z = 0
    z = acd - r + math.sqrt(r*r - rr)

    return z  # corneal surface position relative to pupil plane



def calc_distance_laser_corner(laser_position: PupilCoord) -> float:
    cornea_height = calc_anterior_cornea_z_position(x=laser_position.x,
                                                    y=laser_position.y)
    return laser_position.z - cornea_height