"""LEGACY LOAD COMPATIBILITY ONLY — the dark-frame feature was removed
2026-08-21 (production fits RAW frames; the dark level and read noise come
from ccd_characteristics). Nothing in production constructs these classes.

They stay because old saved files contain them and unpickling needs the
classes at this exact module path:
  * ImageStatistics — the dark_image inside SystemState of older scans,
  * BackgroundImage — the on-disk format of the removed "Save darks" button
    (e.g. Data/2025/2025-6-15/dark_noise_*.pkl/.h5).
Both are also registered in known_dataclasses_lookup so the HDF5 loader can
rebuild them. Two classes in one file, against the one-dataclass-per-file
rule, because this is a frozen legacy file format — do not import from here
in new code.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ImageStatistics:
    mean_image: np.ndarray
    std_image: np.ndarray
    # Defaults so the h5 loader can rebuild files from before these fields
    # existed (e.g. the 2025-6-15 dark files carry no median_image).
    median_image: np.ndarray | None = None
    n: int = 0


@dataclass
class BackgroundImage:
    dark_image: ImageStatistics | None = None
