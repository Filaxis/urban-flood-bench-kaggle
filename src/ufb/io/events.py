from dataclasses import dataclass
from pathlib import Path
from typing import Dict

def index_event_folders(model_split_root: Path) -> Dict[int, Path]:
    """
    model_split_root: Models/Model_{id}/train OR /test
    returns: {event_id_int: path_to_event_folder}
    """
    mapping: Dict[int, Path] = {}
    for p in sorted(model_split_root.glob("event_*")):
        # Accept event_5, event_05, event_069, etc.
        suffix = p.name.split("_", 1)[1]
        try:
            eid = int(suffix)
        except ValueError:
            continue
        mapping[eid] = p
    return mapping
