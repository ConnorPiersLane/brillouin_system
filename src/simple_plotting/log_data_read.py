from __future__ import annotations

import ast
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

NAME_LINE_RE = re.compile(r"^\s*([^:\n][^:\n]*):\s*$")
FIELD_START_RE = re.compile(r"^\s*([^:\n][^:\n]*):\s*(.*)$")
NP_FLOAT_RE = re.compile(r"np\.float64\((.*?)\)")


def clean_value_text(text: str) -> str:
    """Normalize numpy-style wrappers and collapse extra whitespace."""
    text = NP_FLOAT_RE.sub(r"\1", text)
    return " ".join(text.split())


def parse_bracket_list(text: str) -> list[float]:
    """
    Parse list text like:
    [1 2 3
     4 5 6]
    or
    [1, 2, 3]
    """
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []

    parts = re.split(r"[\s,]+", inner)
    return [float(p) for p in parts if p]


def parse_tuple(text: str) -> tuple[Any, ...]:
    """
    Parse tuple text like:
    (0.1, -0.2, 5.4)
    """
    cleaned = clean_value_text(text)
    try:
        value = ast.literal_eval(cleaned)
        if isinstance(value, tuple):
            return value
        return (value,)
    except Exception:
        inner = cleaned.strip()[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
        return tuple(out)


def parse_scalar(text: str) -> Any:
    cleaned = clean_value_text(text)

    for caster in (int, float):
        try:
            return caster(cleaned)
        except ValueError:
            pass

    return cleaned


def parse_value(text: str) -> Any:
    text = text.strip()

    if text.startswith("[") and text.endswith("]"):
        return parse_bracket_list(text)

    if text.startswith("(") and text.endswith(")"):
        return parse_tuple(text)

    return parse_scalar(text)


def collect_multiline_value(
    lines: list[str], start_index: int, first_value: str
) -> tuple[str, int]:
    """
    Collect a possibly multi-line value until [] and () are balanced.
    Returns:
        (full_value_text, next_index)
    """
    value_lines = [first_value.rstrip()]
    i = start_index + 1

    bracket_balance = first_value.count("[") - first_value.count("]")
    paren_balance = first_value.count("(") - first_value.count(")")

    while i < len(lines) and (bracket_balance > 0 or paren_balance > 0):
        line = lines[i].rstrip()
        value_lines.append(line)
        bracket_balance += line.count("[") - line.count("]")
        paren_balance += line.count("(") - line.count(")")
        i += 1

    full_value = " ".join(v.strip() for v in value_lines)
    return full_value, i


def add_value(record: dict[str, Any], key: str, value: Any) -> None:
    """
    Add a key/value to the record.
    If the key appears multiple times, collect all values into a list.

    Example:
        daq_ts: [1 2 3]
        daq_ts: [4 5 6]

    becomes:
        "daq_ts": [
            [1, 2, 3],
            [4, 5, 6]
        ]
    """
    if key in record:
        if isinstance(record[key], list) and key in record.get("_repeated_keys", set()):
            record[key].append(value)
        else:
            record[key] = [record[key], value]
            record.setdefault("_repeated_keys", set()).add(key)
    else:
        record[key] = value


def parse_record(lines: list[str]) -> dict[str, Any]:
    """
    Parse one record block. The first line is the record name, e.g.
        Did not find Reflection Plane:
    """
    if not lines:
        return {}

    name_match = NAME_LINE_RE.match(lines[0])
    if not name_match:
        raise ValueError(f"First line is not a valid name line: {lines[0]!r}")

    record: dict[str, Any] = {"name": name_match.group(1).strip()}

    i = 1
    while i < len(lines):
        raw = lines[i].rstrip()
        if not raw.strip():
            i += 1
            continue

        field_match = FIELD_START_RE.match(raw)
        if not field_match:
            i += 1
            continue

        key = field_match.group(1).strip()
        first_value = field_match.group(2).strip()

        full_value, next_i = collect_multiline_value(lines, i, first_value)
        value = parse_value(full_value)
        add_value(record, key, value)
        i = next_i

    # remove helper key if used
    record.pop("_repeated_keys", None)
    return record


def split_into_records(text: str) -> list[list[str]]:
    """
    Split file text into records.

    A new record starts when a line looks like:
        Some name:
    and the next non-empty line looks like a field:
        key: value
    """
    lines = text.splitlines()
    records: list[list[str]] = []
    current: list[str] = []

    def is_name_line(idx: int) -> bool:
        if idx >= len(lines):
            return False
        if not NAME_LINE_RE.match(lines[idx]):
            return False

        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        return j < len(lines) and FIELD_START_RE.match(lines[j]) is not None

    for i, line in enumerate(lines):
        if is_name_line(i):
            if current:
                records.append(current)
                current = []
        current.append(line)

    if current:
        records.append(current)

    return [r for r in records if r and NAME_LINE_RE.match(r[0])]


def parse_file(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    record_lines = split_into_records(text)
    return [parse_record(block) for block in record_lines]


def make_json_safe(obj: Any) -> Any:
    """Convert tuples/sets/etc. into JSON-safe types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [make_json_safe(v) for v in obj]
    return obj


def parse_many(pattern: str) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for path in glob.glob(pattern):
        results[path] = parse_file(path)
    return results


if __name__ == "__main__":
    # Change this to match your files
    pattern = "*.txt"

    data = parse_many(pattern)
    safe_data = make_json_safe(data)
    log = data['log.txt']  # your parsed data
    rows = []
    for entry in log:
        try:
            # ID
            scan_id = entry.get("name", "unknown")

            # Laser position
            lp = entry.get("laser position", (None, None, None))
            x, y = lp[0], lp[1]

            # Compute geometry
            radius = math.sqrt(x ** 2 + y ** 2) if x is not None else None
            angle = math.degrees(math.atan2(y, x)) if x is not None else None

            # DAQ values → take max
            daq_values = entry.get("daq_values", [])
            max_daq = max(daq_values) if daq_values else None

            # is_found (based on your naming)
            is_found = "Did not find" not in scan_id

            rows.append({
                "ID": scan_id,
                "Angle": angle,
                "Radius": radius,
                "is_found": is_found,
                "max daq signal [V]": max_daq
            })

        except Exception as e:
            print(f"Skipping entry: {e}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to Excel
    output_path = "log_analysis.xlsx"
    df.to_excel(output_path, index=False)

    print(f"Saved Excel to: {output_path}")