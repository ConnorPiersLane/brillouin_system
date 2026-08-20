"""Regression test: scans loaded from HDF5 carry sline_rows as a numpy
array, whose truth value is ambiguous. fit_axial_scan crashed on them with
"ValueError: The truth value of an array..." (Axial Scan Manager,
2026-08-20). stored_sline_rows() normalizes every storage form.
"""
from types import SimpleNamespace

import numpy as np

from brillouin_system.my_dataclasses.human_interface_measurements import (
    stored_sline_rows,
)


def test_plain_list():
    scan = SimpleNamespace(sline_rows=[5, 6, 7])
    assert stored_sline_rows(scan) == [5, 6, 7]


def test_numpy_array():
    scan = SimpleNamespace(sline_rows=np.array([5, 6, 7]))
    rows = stored_sline_rows(scan)
    assert rows == [5, 6, 7]
    assert all(isinstance(r, int) for r in rows)


def test_none_and_missing():
    assert stored_sline_rows(SimpleNamespace(sline_rows=None)) is None
    assert stored_sline_rows(SimpleNamespace()) is None


def test_empty_forms():
    assert stored_sline_rows(SimpleNamespace(sline_rows=[])) is None
    assert stored_sline_rows(
        SimpleNamespace(sline_rows=np.array([], dtype=int))) is None
