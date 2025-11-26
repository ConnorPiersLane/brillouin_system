from dataclasses import dataclass


@dataclass
class Ellipse2D:
    cx: float       # center of the fitted image
    cy: float
    major: float
    minor: float
    angle_deg: float
    cx_original: float | None = None  # center of the original frame
    cy_original: float | None = None

    @property
    def center(self) -> tuple[float, float]: return (self.cx, self.cy)

    @property
    def center_original(self) -> tuple[float, float]: return (self.cx_original, self.cy_original)

    @property
    def axes(self) -> tuple[float, float]: return (self.major, self.minor)

    @property
    def axis_ratio(self) -> float: return self.major / max(self.minor, 1e-9)

