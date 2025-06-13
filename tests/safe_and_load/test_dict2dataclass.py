
# Dummy dataclasses for test
from dataclasses import dataclass

import numpy as np

from brillouin_system.saving_and_loading.dict2dataclass import dict_to_dataclass_tree


@dataclass
class InnerClass:
    values: np.ndarray

@dataclass
class Outer:
    inner_class: InnerClass
    calibration_data: str | None

def test_rehydrate():
    raw = {
        "outer": {
            "inner_class": {
                "values": [1, 2, 3]
            },
            "calibration_data": "__NONE__"
        }
    }

    known = {
        "Outer": Outer,
        "Inner": InnerClass,
    }

    result = dict_to_dataclass_tree(raw["outer"], "outer", known_classes=known)
    print("✅ Result:", result)

    assert isinstance(result, Outer)
    assert isinstance(result.inner_class, InnerClass)
    assert isinstance(result.inner_class.values, np.ndarray)
    assert result.calibration_data is None



if __name__ == "__main__":
    test_rehydrate()
