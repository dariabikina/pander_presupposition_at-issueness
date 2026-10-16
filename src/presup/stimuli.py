import csv
from pathlib import Path

from presup.paths import STIMULI

NUMERIC = {"baseline_avg": float, "baseline_gemini_3_7_flash": float, "words": int, "chars": int}


def load_stimuli(path: Path = STIMULI, required: tuple[str, ...] = ()) -> list[dict]:
    """One dict per prompt, with sample_id = item_id__format__condition."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path.name} is missing columns: {missing}")
        records = []
        for row in reader:
            # empty cells are None, numbers are numbers
            record = {k: (NUMERIC[k](v) if k in NUMERIC else v) if v != "" else None
                      for k, v in row.items()}
            record["sample_id"] = f"{record['item_id']}__{record['format']}__{record['condition']}"
            records.append(record)
    if len({r["sample_id"] for r in records}) != len(records):
        raise SystemExit(f"duplicate sample_ids in {path.name}")
    return records
