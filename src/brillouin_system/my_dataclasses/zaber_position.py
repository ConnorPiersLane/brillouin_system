from dataclasses import dataclass


@dataclass
class ZaberPosition:
    x: float
    y: float
    z: float


def generate_zaber_positions(
    axis: str,
    start: float,
    step: float,
    n: int,
    fixed_positions: dict[str, float] = None
) -> list[ZaberPosition]:
    """
    Generates a list of ZaberPosition instances sweeping over the specified axis.

    Args:
        axis: The axis to vary ('x', 'y', or 'z').
        start: Starting position in micrometers.
        step: Step size in micrometers.
        n: Number of steps to generate.
        fixed_positions: A dict with fixed positions for other axes.

    Returns:
        A list of ZaberPosition instances.
    """
    if fixed_positions is None:
        fixed_positions = {}

    # Ensure all axes are covered
    all_axes = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    all_axes.update(fixed_positions)  # override defaults with provided values

    positions = []
    for i in range(n):
        current = start + i * step
        pos_kwargs = all_axes.copy()
        pos_kwargs[axis] = current
        positions.append(ZaberPosition(**pos_kwargs))

    return positions
