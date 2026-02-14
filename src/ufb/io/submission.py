import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple, Optional

from ufb.io.write_plan import WritePlan

def write_submission_from_blocks(
    write_plan: WritePlan,
    out_csv: Path,
    predict_event: Callable[[int, int], Dict[Tuple[int, int], List[float]]],
):
    """
    predict_event(model_id, event_id) returns a dict:
      key: (node_type, node_id)
      value: list/array of predicted water levels length H for that node
    """

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"])

        current_event = None
        cache = None  # predictions for current (model,event)

        for block in write_plan.blocks:
            evt = (block.model_id, block.event_id)
            if evt != current_event:
                cache = predict_event(block.model_id, block.event_id)
                current_event = evt

            assert cache is not None
            series = cache[(block.node_type, block.node_id)]
            if len(series) != block.length:
                raise ValueError(f"Prediction length mismatch for {evt} type={block.node_type} node={block.node_id}: "
                                 f"expected {block.length}, got {len(series)}")

            # Emit rows in exact template order
            for k in range(block.length):
                row_id = block.start_row + k
                w.writerow([row_id, block.model_id, block.event_id, block.node_type, block.node_id, float(series[k])])
