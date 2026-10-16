"""Does an item's prior credence predict how much the manipulation moves it?

analyse.py tests a linear trend in the prior. The hypothesis here is a peak at prior 0.5 (the
model is least sure, so it moves most), so there are two predictors, prior and uncertainty =
0.5 - |prior - 0.5|, and two outcomes, the signed per-item effect and its absolute value.

A noisier item has a larger |effect| even with no true effect, and the mid-prior items are
noisier, so |effect| is also residualised on the item's sample-to-sample SD and re-tested
(splithalf.py is the cleaner check).

Spearman rho with a permutation p, BH-FDR across all tests per model, and tercile means.

    uv run python analysis/moderator.py
"""

from __future__ import annotations

import argparse
import statistics

from presup.design import ENCODING_PAIRS, ENCODINGS, cell_means
from presup.paths import MODELS, pilot_csv, read_csv
from presup.stats import (Q, as_float, benjamini_hochberg, bootstrap_statistic, ols_slope,
                          permutation_p_rho, spearman)


def per_item_effects(means: dict, fmt: str, label: str) -> dict[str, float]:
    """Per-item effect for one dimension in one format."""
    items = sorted({key[0] for key in means})
    out: dict[str, float] = {}
    for item in items:
        if label == "nai - ai":
            parts = [means.get((item, fmt, f"NAI_{e}")) for e in ENCODINGS], \
                    [means.get((item, fmt, f"AI_{e}")) for e in ENCODINGS]
            pairs = [(a, b) for a, b in zip(*parts) if a is not None and b is not None]
            if pairs:
                out[item] = statistics.fmean(a - b for a, b in pairs)
        elif label == "instr - conv":
            cells = [f"{lvl}_{e}" for lvl in ("NAI", "AI") for e in ENCODINGS]
            pairs = [(means.get((item, "instr", c)), means.get((item, "conv", c))) for c in cells]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if pairs:
                out[item] = statistics.fmean(a - b for a, b in pairs)
        else:
            _, left, right = next(p for p in ENCODING_PAIRS if p[0] == label)
            a, b = means.get((item, fmt, left)), means.get((item, fmt, right))
            if a is not None and b is not None:
                out[item] = a - b
    return out


def item_noise(rows: list[dict]) -> dict[str, float]:
    """Each item's mean SD of credence across samples of the same prompt."""
    cells: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = as_float(row.get("consensus_credence"))
        if value is not None:
            cells.setdefault((row["item_id"], row["format"], row["condition"]), []).append(value)
    spreads: dict[str, list[float]] = {}
    for (item, _, _), values in cells.items():
        if len(values) >= 2:
            spreads.setdefault(item, []).append(statistics.stdev(values))
    return {item: statistics.fmean(values) for item, values in spreads.items()}


def residualise(ys: list[float], covariate: list[float]) -> list[float]:
    """y minus its OLS fit on the covariate."""
    slope = ols_slope(covariate, ys)
    y_mean, c_mean = statistics.fmean(ys), statistics.fmean(covariate)
    intercept = y_mean - slope * c_mean
    return [y - (intercept + slope * c) for y, c in zip(ys, covariate)]


def terciles(xs: list[float], ys: list[float]) -> list[tuple[str, float, int]]:
    """Mean outcome in each third of the prior range."""
    bands = (("0.00-0.33", 0.0, 1 / 3), ("0.33-0.67", 1 / 3, 2 / 3), ("0.67-1.00", 2 / 3, 1.01))
    out = []
    for name, low, high in bands:
        vals = [y for x, y in zip(xs, ys) if low <= x < high]
        out.append((name, statistics.fmean(vals) if vals else float("nan"), len(vals)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    dimensions = [p[0] for p in ENCODING_PAIRS] + ["nai - ai", "instr - conv"]

    for model in args.models:
        rows = [r for r in read_csv(pilot_csv(model)) if r["error"] == ""]
        means = cell_means(rows, lambda r: as_float(r.get("consensus_credence")))
        baselines = {r["item_id"]: v for r in rows
                     if (v := as_float(r.get("baseline_avg"))) is not None}
        noise = item_noise(rows)

        print("=" * 104)
        print(f"{model}   ({len(rows)} answered rows)")
        print("=" * 104)

        tests = []
        for dim in dimensions:
            formats = ["pooled"] if dim == "instr - conv" else ["conv", "instr"]
            for fmt in formats:
                effects = per_item_effects(means, "conv" if fmt == "pooled" else fmt, dim)
                items = [i for i in effects if i in baselines and i in noise]
                if len(items) < 6:
                    continue
                prior = [baselines[i] for i in items]
                signed = [effects[i] for i in items]
                absolute = [abs(effects[i]) for i in items]
                uncert = [0.5 - abs(p - 0.5) for p in prior]
                abs_resid = residualise(absolute, [noise[i] for i in items])
                outcomes = (("signed", signed), ("absolute", absolute),
                            ("abs|noise", abs_resid))
                for outcome_name, outcome in outcomes:
                    for pred_name, pred in (("prior", prior), ("uncertainty", uncert)):
                        rho = spearman(pred, outcome)
                        pairs = list(zip(pred, outcome))
                        lo, hi = bootstrap_statistic(pairs, spearman)
                        tests.append({
                            "dim": dim, "fmt": fmt, "outcome": outcome_name, "pred": pred_name,
                            "rho": rho, "lo": lo, "hi": hi,
                            "p": permutation_p_rho(pred, outcome), "n": len(items),
                            "slope": ols_slope(pred, outcome),
                            "bands": terciles(prior, outcome),
                        })

        survivors = benjamini_hochberg([t["p"] for t in tests])
        print(f"\n{len(tests)} tests; BH-FDR across all of them at q<{Q}\n")
        header = (f"  {'dimension':<17}{'fmt':<7}{'outcome':<10}{'predictor':<13}"
                  f"{'rho':>7}{'95% CI':>18}{'p':>8}{'n':>4}  verdict")
        print(header)
        hits = []
        for test, survives in zip(tests, survivors):
            mark = ("**" if test["p"] < 0.01 else "*") if survives else (
                "(raw only)" if test["p"] < 0.05 else "ns")
            if survives:
                hits.append(test)
            print(f"  {test['dim']:<17}{test['fmt']:<7}{test['outcome']:<10}{test['pred']:<13}"
                  f"{test['rho']:+7.3f}  [{test['lo']:+.3f},{test['hi']:+.3f}]"
                  f"{test['p']:>8.4f}{test['n']:>4}  {mark}")

        print(f"\n  -> {len(hits)} of {len(tests)} survive BH at q<{Q}")

        print("\n  TERCILE MEANS of the absolute effect, by prior band")
        print(f"  {'dimension':<17}{'fmt':<7}{'0.00-0.33':>14}{'0.33-0.67':>14}{'0.67-1.00':>14}")
        for test in tests:
            if test["outcome"] != "absolute" or test["pred"] != "prior":
                continue
            cells = "".join(f"{m:>10.3f} n{n:<3}" if m == m else f"{'-':>14}"
                            for _, m, n in test["bands"])
            print(f"  {test['dim']:<17}{test['fmt']:<7}{cells}")
        print()


if __name__ == "__main__":
    main()
