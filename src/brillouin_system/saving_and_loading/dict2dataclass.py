import inspect
import sys
from dataclasses import is_dataclass, fields
from typing import get_origin, get_args


from .safe_and_load_dict2hdf5 import _NONE_MARKER


def snake_to_pascal(snake: str) -> str:
    return ''.join(part.capitalize() for part in snake.split('_'))


def dict_to_dataclass_tree(data, name_hint=None, known_classes=None):
    """
    Recursively convert dicts (with potential name hints) to matching dataclasses.
    Lists/tuples are preserved as lists if the dataclass field is list[...] type.
    "__NONE__" is converted to None.
    """
    if known_classes is None:
        known_classes = {
            cls.__name__: cls
            for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
            if is_dataclass(cls)
        }

    if data == _NONE_MARKER:
        return None

    # Default for free-floating lists (outside dataclass fields)
    if isinstance(data, (list, tuple)):
        return [dict_to_dataclass_tree(item, None, known_classes) for item in data]

    if isinstance(data, dict):
        cls = None
        if name_hint:
            class_name = snake_to_pascal(name_hint)
            cls = known_classes.get(class_name)

        # Fallback to structural match
        if not cls:
            for candidate_cls in known_classes.values():
                if is_dataclass(candidate_cls):
                    field_names = {f.name for f in fields(candidate_cls)}
                    if set(data.keys()) <= field_names:
                        cls = candidate_cls
                        break

        # Reconstruct dataclass
        if cls:
            kwargs = {}
            for f in fields(cls):
                if f.name not in data:
                    continue

                value = data[f.name]
                origin = get_origin(f.type) or f.type
                args = get_args(f.type)

                # Case: list[T] or tuple[T]
                if isinstance(value, (list, tuple)) and origin in (list, tuple) and args:
                    item_type = args[0]
                    if is_dataclass(item_type):
                        value = [
                            dict_to_dataclass_tree(v, item_type.__name__.lower(), known_classes)
                            for v in value
                        ]
                    else:
                        value = [dict_to_dataclass_tree(v, f.name, known_classes) for v in value]

                # Case: nested dataclass
                elif isinstance(value, dict) and is_dataclass(origin):
                    value = dict_to_dataclass_tree(value, origin.__name__.lower(), known_classes)

                else:
                    value = dict_to_dataclass_tree(value, f.name, known_classes)

                kwargs[f.name] = value

            return cls(**kwargs)

        # Fallback: dict with converted children
        return {k: dict_to_dataclass_tree(v, k, known_classes) for k, v in data.items()}

    return data
