"""Reading the challenge measure out of a pilot CSV, plus rater agreement.

chal_open_* is the judge with the full prompt, chal_blind_* with the claim sentence redacted
(see blinding.py); blind is primary for any comparison between framings. Each has two raters.
<prefix>_score is the mean of the non-null raters (0, 0.5 or 1). Two binary raters disagree most
when the true rate is near 0.5, which is where the cells of interest are, so dropping
disagreements would thin exactly those cells. <prefix>_consensus (both agree, else null) is kept
as a sensitivity check.

NO_CLAIM has no claim to challenge, so its rate is the judge's false-positive baseline and every
rate is printed next to it.
"""

from __future__ import annotations

import statistics

from presup.stats import as_float

VARIANTS = ("blind", "open")
PRIMARY_VARIANT = "blind"
RATERS = (1, 2)


def prefix(variant: str) -> str:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown challenge instrument {variant!r}; expected one of {VARIANTS}")
    return f"chal_{variant}"


def _as_bool(value: object) -> bool | None:
    # blank or unrecognised -> None
    return {"true": True, "false": False}.get(str(value).strip().lower())


def available_variants(rows: list[dict]) -> tuple[str, ...]:
    """Which instruments this CSV holds, in reporting order."""
    if not rows:
        return ()
    return tuple(v for v in VARIANTS if f"{prefix(v)}_score" in rows[0])


def source_note(rows: list[dict]) -> str:
    """One line naming which measure the numbers below came from."""
    if not rows:
        return "no rows"
    present = available_variants(rows)
    if not present:
        raise SystemExit(
            "this CSV has no challenge columns at all: expected "
            f"{[f'{prefix(v)}_score' for v in VARIANTS]}"
        )
    return (f"instruments {list(present)}, each = mean of the non-null raters in {{0, 0.5, 1}}; "
            f"primary = {PRIMARY_VARIANT} (frame-blind); `_consensus` is the stricter "
            "both-must-agree rule")


def score(row: dict, variant: str = PRIMARY_VARIANT) -> float | None:
    """0, 0.5 or 1, or None if no rater produced a verdict."""
    return as_float(row.get(f"{prefix(variant)}_score"))


def consensus(row: dict, variant: str = PRIMARY_VARIANT) -> bool | None:
    """Both raters agreed, else None."""
    return _as_bool(row.get(f"{prefix(variant)}_consensus"))


def raters(row: dict, variant: str = PRIMARY_VARIANT) -> list[bool | None]:
    """Each rater's own verdict, in order."""
    columns = [f"{prefix(variant)}{index}_challenged" for index in RATERS]
    return [_as_bool(row[column]) for column in columns if column in row]


def _kappa(pairs: list[tuple[bool, bool]]) -> tuple[float | None, float | None]:
    """(percent agreement, Cohen's kappa); kappa is None when one side never varied."""
    if not pairs:
        return (None, None)
    observed = statistics.fmean([1.0 if a == b else 0.0 for a, b in pairs])
    p1 = statistics.fmean([1.0 if a else 0.0 for a, _ in pairs])
    p2 = statistics.fmean([1.0 if b else 0.0 for _, b in pairs])
    chance = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - chance) / (1 - chance) if abs(1 - chance) > 1e-12 else None
    return (observed, kappa)


def agreement(rows: list[dict], variant: str = PRIMARY_VARIANT) -> dict:
    """Inter-rater agreement for one instrument; rows missing a rater are counted apart."""
    pairs = [r for r in (raters(row, variant) for row in rows) if len(r) == len(RATERS)]
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    observed, kappa = _kappa(both)
    return {"what": f"{variant} raters 1 vs 2", "n_rows": len(rows), "n_pairs": len(both),
            "n_missing": len(pairs) - len(both),
            "percent_agreement": observed, "kappa": kappa}


def frame_contamination(rows: list[dict]) -> dict:
    """Blinded vs unblinded verdicts on the same response by the same rater."""
    pairs: list[tuple[bool, bool]] = []
    deltas: list[float] = []
    for row in rows:
        for blind, open_ in zip(raters(row, "blind"), raters(row, "open")):
            if blind is not None and open_ is not None:
                pairs.append((blind, open_))
        blind_score, open_score = score(row, "blind"), score(row, "open")
        if blind_score is not None and open_score is not None:
            deltas.append(blind_score - open_score)
    observed, kappa = _kappa(pairs)
    return {"what": "blind vs open, same rater and response", "n_rows": len(rows),
            "n_pairs": len(pairs), "n_missing": 0,
            "percent_agreement": observed, "kappa": kappa,
            "n_scored": len(deltas),
            "mean_delta": statistics.fmean(deltas) if deltas else None}


def format_agreement(stats: dict) -> str:
    """An agreement dict as one printed line."""
    if not stats["n_pairs"]:
        return f"{stats['what']}: 0 of {stats['n_rows']} rows usable, no statistic available"
    kappa = "n/a (one side never varied)" if stats["kappa"] is None else f"{stats['kappa']:+.3f}"
    line = (f"{stats['what']}: agreement {stats['percent_agreement']:.0%} on "
            f"{stats['n_pairs']} rated pair(s) from {stats['n_rows']} rows; kappa {kappa}")
    if stats.get("n_missing"):
        line += f"; {stats['n_missing']} row(s) missing a rater"
    if stats.get("mean_delta") is not None:
        line += (f"; mean blinded - unblinded score {stats['mean_delta']:+.3f} "
                 f"over {stats['n_scored']} responses")
    return line


def baseline_corrected(rate: float, baseline: float | None) -> str:
    """A rate read against the NO_CLAIM false-positive rate for the same model and format."""
    if baseline is None:
        return "no NO_CLAIM baseline in this slice"
    return f"{rate - baseline:+.0%} vs baseline {baseline:.0%}"
