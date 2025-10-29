from dataclasses import dataclass


@dataclass
class Ellipse2D:
    cx: float
    cy: float
    major: float
    minor: float
    angle_deg: float

    @property
    def center(self) -> tuple[float, float]: return (self.cx, self.cy)

    @property
    def axes(self) -> tuple[float, float]: return (self.major, self.minor)

    @property
    def axis_ratio(self) -> float: return self.major / max(self.minor, 1e-9)