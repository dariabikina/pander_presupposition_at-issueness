"""Presupposition contrasts split by the proposition's prior credence band.

Prints NAI_PRESUP and AI_PRESUP minus every other cell within each third of the prior range,
with n next to every number since the bands are small. With 5 items the smallest possible
sign-flip p is 0.0625, so there are no p-values here; a bootstrap interval that excludes zero
is marked, as a description only.

    uv run python analysis/bands.py --models anthropic/claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import statistics

from presup.design import CELL_ORDER, cell_means
from presup.paths import MODELS, pilot_csv, read_csv
from presup.stats import as_float, bootstrap

# half-open thirds of the prior range
BANDS = (("0.00-0.33", 0.0, 1 / 3), ("0.33-0.67", 1 / 3, 2 / 3), ("0.67-1.00", 2 / 3, 1.01))
PRESUP = ("NAI_PRESUP", "AI_PRESUP")


def priors(rows: list[dict]) -> dict[str, float]:
    out = {}
    for row in rows:
        value = as_float(row.get("baseline_avg"))
        if value is not None:
            out[row["item_id"]] = value
    return out


def band_of(prior: float) -> str | None:
    for name, low, high in BANDS:
        if low <= prior < high:
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    for model in args.models:
        bands(model)


def bands(model: str) -> None:
    rows = [r for r in read_csv(pilot_csv(model)) if not r.get("error")]
    prior = priors(rows)
    means = cell_means(rows, lambda r: as_float(r.get("consensus_credence")))
    items = sorted({k[0] for k in means})

    print("=" * 100)
    print(f"{model}   presupposition contrasts by prior credence band")
    print(f"{len(rows)} answered rows, {len(items)} items")
    counts = {name: sum(1 for i in items if band_of(prior[i]) == name) for name, _, _ in BANDS}
    print("  items per band: " + ", ".join(f"{k} n={v}" for k, v in counts.items()))
    print("  smallest attainable sign-flip p: " + ", ".join(
        f"{k} {2 / 2 ** v:.3f}" for k, v in counts.items()))
    print("=" * 100)

    flagged = []
    for target in PRESUP:
        others = [c for c in CELL_ORDER if c != target]
        for fmt in ("conv", "instr"):
            print(f"\n{target} minus X, {fmt}   (positive = presupposing raised credence)")
            print(f"  {'vs.':<12}" + "".join(f"{b[0]:>18}" for b in BANDS))
            for other in others:
                cells = []
                for name, _, _ in BANDS:
                    diffs = []
                    for item in items:
                        a = means.get((item, fmt, target))
                        b = means.get((item, fmt, other))
                        if a is None or b is None or band_of(prior[item]) != name:
                            continue
                        diffs.append(a - b)
                    if not diffs:
                        cells.append(f"{'-':>18}")
                        continue
                    m = statistics.fmean(diffs)
                    lo, hi = bootstrap(diffs) if len(diffs) > 1 else (float("nan"),) * 2
                    mark = "*" if lo == lo and (lo > 0 or hi < 0) else " "
                    if mark == "*":
                        flagged.append((target, fmt, other, name, m, lo, hi, len(diffs)))
                    cells.append(f"{m:+.3f} n{len(diffs)}{mark:>2}".rjust(18))
                tag = other + (" †" if other == "NO_CLAIM" else "")
                print(f"  {tag:<12}" + "".join(cells))

    print("\n  * bootstrap 95% interval excludes zero (descriptive; not corrected for the grid)")
    print("  † NO_CLAIM credence is recoverable from only ~half of responses, so these rows rest")
    print("    on a different, non-random subset of items.")
    if flagged:
        print(f"\n  cells whose interval excludes zero ({len(flagged)} of "
              f"{len(PRESUP) * 2 * (len(CELL_ORDER) - 1) * len(BANDS)}):")
        for t, f, o, b, m, lo, hi, n in sorted(flagged, key=lambda x: -abs(x[4])):
            print(f"    {t:<11}{f:<6}- {o:<11}{b:<11}{m:+.3f}  [{lo:+.3f},{hi:+.3f}]  n={n}")


if __name__ == "__main__":
    main()
