import statistics
from collections import defaultdict

# NO_CLAIM, then the four framings at each at-issueness level
AT_ISSUENESS = ("NAI", "AI")
ENCODINGS = ("ASSERT", "HEDGE", "COMMIT", "PRESUP")
CELL_ORDER = ("NO_CLAIM", *(f"{level}_{encoding}" for level in AT_ISSUENESS for encoding in ENCODINGS))
FORMATS = ("conv", "instr")

# per-item encoding contrasts, at the not-at-issue level
ENCODING_PAIRS = (
    ("presup - assert", "NAI_PRESUP", "NAI_ASSERT"),
    ("presup - commit", "NAI_PRESUP", "NAI_COMMIT"),
    ("hedge - presup", "NAI_HEDGE", "NAI_PRESUP"),
    ("hedge - commit", "NAI_HEDGE", "NAI_COMMIT"),
)


def cell_means(rows: list[dict], value_of) -> dict[tuple[str, str, str], float]:
    """Mean of value_of(row) per (item, format, condition), over samples."""
    buckets = defaultdict(list)
    for row in rows:
        value = value_of(row)
        if value is not None:
            buckets[(row["item_id"], row["format"], row["condition"])].append(value)
    return {key: statistics.fmean(values) for key, values in buckets.items()}


def paired(means: dict, fmt: str, left: str, right: str) -> list[tuple[str, float]]:
    """Per-item left minus right, for items that have both cells."""
    return paired_formats(means, fmt, left, fmt, right)


def paired_formats(means: dict, fmt_left: str, left: str,
                   fmt_right: str, right: str) -> list[tuple[str, float]]:
    out = []
    for item in sorted({key[0] for key in means}):
        a, b = means.get((item, fmt_left, left)), means.get((item, fmt_right, right))
        if a is not None and b is not None:
            out.append((item, a - b))
    return out


def paired_combination(means: dict, fmt: str, weights: dict[str, float]) -> list[tuple[str, float]]:
    """Per-item weighted sum of cells, e.g. a main effect or an interaction.

    An item counts only if it has every cell in `weights`.
    """
    out = []
    for item in sorted({key[0] for key in means}):
        if any(means.get((item, fmt, c)) is None for c in weights):
            continue
        out.append((item, sum(w * means[(item, fmt, c)] for c, w in weights.items())))
    return out
