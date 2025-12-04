from dataclasses import dataclass

@dataclass
class RigCoord:
    x: float
    y: float
    z: float

@dataclass
class PupilCoord:
    x: float
    y: float
    z: float




class RigPupilTransform:
    """
    Coordinate transform between rig coordinates and pupil coordinates.

    d = pupil center expressed in rig coordinates
    Rotation = 180° around rig Y-axis
        R = diag([-1, 1, -1])
    """

    def __init__(self, pupil_center: RigCoord):
        # store translation
        self.d = pupil_center
        # rotation matrix entries for 180° rotation around Y
        self.Rx = -1
        self.Ry =  1
        self.Rz = -1

    # -------------------------------------------------------------
    # Rig → Pupil
    # -------------------------------------------------------------
    def rig_to_pupil(self, r: RigCoord) -> PupilCoord:
        """
        Convert a point from rig coordinates to pupil coordinates:
            x_p = R * (x_r - d)
        """
        return PupilCoord(
            x = self.Rx * (r.x - self.d.x),
            y = self.Ry * (r.y - self.d.y),
            z = self.Rz * (r.z - self.d.z),
        )

    # -------------------------------------------------------------
    # Pupil → Rig
    # -------------------------------------------------------------
    def pupil_to_rig(self, p: PupilCoord) -> RigCoord:
        """
        Convert a point from pupil coordinates to rig coordinates:
            x_r = d + R * x_p
        """
        return RigCoord(
            x = self.d.x + self.Rx * p.x,
            y = self.d.y + self.Ry * p.y,
            z = self.d.z + self.Rz * p.z,
        )

if __name__ == "__main__":

    # pupil center in rig coordinates
    d = RigCoord(1, 1, 2)

    T = RigPupilTransform(d)

    # a point in rig coordinates
    r = RigCoord(0,0,1)

    # convert to pupil coordinates
    p = T.rig_to_pupil(r)
    print(p)

    # convert back
    r2 = T.pupil_to_rig(p)
    print(r2)
