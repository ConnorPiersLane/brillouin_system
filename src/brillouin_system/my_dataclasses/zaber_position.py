from dataclasses import dataclass


@dataclass
class ZaberPosition:
    x: float | None = None
    y: float | None = None
    z: float | None = None