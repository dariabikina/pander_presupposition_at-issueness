"""Split-half check on moderator.py: does uncertainty predict the size of the true effect?

moderator.py's |effect| is inflated by noise, and the mid-prior items are noisier. Here the
samples are split by parity and each item's effect is computed twice from disjoint samples. With
effect_A = t + e_A and effect_B = t + e_B and independent mean-zero noise,
E[effect_A * effect_B] = t^2, so the product estimates the squared true effect without bias
however noisy the item is. (|effect_A| vs |effect_B| would not work: noise variance is a stable
property of the item.)

Per dimension: Spearman(effect_A, effect_B) on the signed effects as a reliability gate,
Spearman(uncertainty, product) as the moderation test, and the mean product per uncertainty
tercile. Needs >= 6 samples per prompt.

    uv run python analysis/splithalf.py
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from presup.design import ENCODING_PAIRS, ENCODINGS
from presup.paths import MODELS, pilot_csv, read_csv
from presup.stats import (as_float, benjamini_hochberg, bootstrap_statistic, permutation_p_rho,
                          spearman)

MIN_PER_HALF = 2  # minimum samples per half


def half_means(rows: list[dict]) -> dict[tuple[str, str, str, int], float]:
    """Mean credence per (item, format, condition, half); half 1 = odd samples, 0 = even."""
    buckets: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get("consensus_credence"))
        sample = as_float(row.get("sample"))
        if value is None or sample is None:
            continue
        half = int(sample) % 2
        buckets[(row["item_id"], row["format"], row["condition"], half)].append(value)
    return {key: statistics.fmean(values) for key, values in buckets.items()
            if len(values) >= MIN_PER_HALF}


def effects_by_half(means: dict, fmt: str, dim: str, half: int) -> dict[str, float]:
    """Per-item effect for one dimension, from one half of the samples."""
    items = sorted({key[0] for key in means})
    out: dict[str, float] = {}
    for item in items:
        if dim == "nai - ai":
            pairs = [(means.get((item, fmt, f"NAI_{e}", half)),
                      means.get((item, fmt, f"AI_{e}", half))) for e in ENCODINGS]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if pairs:
                out[item] = statistics.fmean(a - b for a, b in pairs)
        elif dim == "instr - conv":
            cells = [f"{lvl}_{e}" for lvl in ("NAI", "AI") for e in ENCODINGS]
            pairs = [(means.get((item, "instr", c, half)), means.get((item, "conv", c, half)))
                     for c in cells]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if pairs:
                out[item] = statistics.fmean(a - b for a, b in pairs)
        else:
            _, left, right = next(p for p in ENCODING_PAIRS if p[0] == dim)
            a = means.get((item, fmt, left, half))
            b = means.get((item, fmt, right, half))
            if a is not None and b is not None:
                out[item] = a - b
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    dimensions = [p[0] for p in ENCODING_PAIRS] + ["nai - ai", "instr - conv"]

    for model in args.models:
        rows = [r for r in read_csv(pilot_csv(model)) if r["error"] == ""]
        per_prompt = defaultdict(set)
        for row in rows:
            per_prompt[row["sample_id"]].add(row["sample"])
        depth = statistics.median(len(v) for v in per_prompt.values())
        means = half_means(rows)
        baselines = {r["item_id"]: v for r in rows
                     if (v := as_float(r.get("baseline_avg"))) is not None}

        print("=" * 100)
        print(f"{model}   ({len(rows)} answered rows, median {depth:.0f} samples per prompt)")
        print("=" * 100)
        if depth < 6:
            print("\n  refusing to split: a split-half estimate needs >=6 samples per prompt so")
            print(f"  each half has >={MIN_PER_HALF}. This run has a median of {depth:.0f}.")
            print("  Rerun run/run_pilot.py with --samples 10 before using this script.\n")
            continue

        tests = []
        for dim in dimensions:
            formats = ["pooled"] if dim == "instr - conv" else ["conv", "instr"]
            for fmt in formats:
                use = "conv" if fmt == "pooled" else fmt
                a = effects_by_half(means, use, dim, 1)
                b = effects_by_half(means, use, dim, 0)
                items = [i for i in a if i in b and i in baselines]
                if len(items) < 6:
                    continue
                sign_a = [a[i] for i in items]
                sign_b = [b[i] for i in items]
                cross = [a[i] * b[i] for i in items]
                uncert = [0.5 - abs(baselines[i] - 0.5) for i in items]
                rel = spearman(sign_a, sign_b)
                rel_lo, rel_hi = bootstrap_statistic(list(zip(sign_a, sign_b)), spearman)
                mod = spearman(uncert, cross)
                mod_lo, mod_hi = bootstrap_statistic(list(zip(uncert, cross)), spearman)
                tests.append({
                    "dim": dim, "fmt": fmt, "n": len(items),
                    "rel": rel, "rel_lo": rel_lo, "rel_hi": rel_hi,
                    "rel_p": permutation_p_rho(sign_a, sign_b),
                    "mod": mod, "mod_lo": mod_lo, "mod_hi": mod_hi,
                    "mod_p": permutation_p_rho(uncert, cross),
                    "bands": [(name, statistics.fmean(v) if v else float("nan"), len(v))
                              for name, v in (
                                  ("low  unc", [c for c, u in zip(cross, uncert) if u < 1/6]),
                                  ("mid  unc", [c for c, u in zip(cross, uncert)
                                                if 1/6 <= u < 1/3]),
                                  ("high unc", [c for c, u in zip(cross, uncert) if u >= 1/3]))],
                })

        rel_ok = benjamini_hochberg([t["rel_p"] for t in tests])
        mod_ok = benjamini_hochberg([t["mod_p"] for t in tests])

        print("\nGATE - signed reliability: Spearman(effect odd, effect even)")
        print("  does the item's effect replicate across independent samples at all")
        print(f"  {'dimension':<17}{'fmt':<7}{'rho':>7}{'95% CI':>19}{'p':>9}{'n':>4}  verdict")
        for test, ok in zip(tests, rel_ok):
            mark = "SIGNAL" if ok else ("(raw only)" if test["rel_p"] < 0.05 else "noise")
            print(f"  {test['dim']:<17}{test['fmt']:<7}{test['rel']:+7.3f}"
                  f"  [{test['rel_lo']:+.3f},{test['rel_hi']:+.3f}]{test['rel_p']:>9.4f}"
                  f"{test['n']:>4}  {mark}")

        print("\nMODERATION - Spearman(uncertainty, effect_odd * effect_even)")
        print("  the product is an unbiased estimate of the squared true effect")
        print(f"  {'dimension':<17}{'fmt':<7}{'rho':>7}{'95% CI':>19}{'p':>9}{'n':>4}  verdict")
        for test, ok, gate in zip(tests, mod_ok, rel_ok):
            if not gate:
                verdict = "gate failed - uninterpretable"
            else:
                verdict = "MODERATED" if ok else "no moderation"
            print(f"  {test['dim']:<17}{test['fmt']:<7}{test['mod']:+7.3f}"
                  f"  [{test['mod_lo']:+.3f},{test['mod_hi']:+.3f}]{test['mod_p']:>9.4f}"
                  f"{test['n']:>4}  {verdict}")

        print("\n  MEAN CROSS-PRODUCT by uncertainty tercile (credence^2; unbiased, can be <0)")
        print(f"  {'dimension':<17}{'fmt':<7}{'low unc':>14}{'mid unc':>14}{'high unc':>14}")
        for test in tests:
            cells = "".join(f"{m:>+10.4f} n{n:<3}" if m == m else f"{'-':>14}"
                            for _, m, n in test["bands"])
            print(f"  {test['dim']:<17}{test['fmt']:<7}{cells}")

        passed = sum(1 for g in rel_ok if g)
        both = sum(1 for g, m in zip(rel_ok, mod_ok) if g and m)
        print(f"\n  -> {passed} of {len(tests)} dimensions have a signed effect that replicates")
        print(f"  -> {both} of those also show uncertainty predicting the squared true effect")
        print()


if __name__ == "__main__":
    main()
