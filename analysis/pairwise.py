"""Every pairwise condition comparison on the credence data, BH-FDR within each family.

The exploratory scan next to analyse.py's planned contrasts: all 36 condition pairs within each
format, and instr - conv within each condition. Everything is paired by item, since items differ
a lot in baseline credence. Each comparison gets a bootstrap 95% CI over items and a two-sided
sign-flip p (17-20 items is too few for a t test); BH rather than Bonferroni because the pairs
share cells and are correlated.

    uv run python analysis/pairwise.py
"""

from __future__ import annotations

import argparse
import itertools
import statistics

from presup.design import CELL_ORDER, FORMATS, cell_means, paired_formats
from presup.paths import MODELS, pilot_csv, read_csv
from presup.stats import Q, as_float, benjamini_hochberg, bootstrap, permutation_p


def stars(p: float, survives: bool) -> str:
    if p != p:
        return ""
    if not survives:
        return "ns" if p >= 0.05 else "(raw only)"
    return "**" if p < 0.01 else "*"


def run_family(title: str, comparisons: list[tuple[str, list[tuple[str, float]]]]) -> int:
    """Print one family of comparisons, BH-corrected within the family."""
    rows = []
    for label, pairs in comparisons:
        values = [value for _, value in pairs]
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        low, high = bootstrap(values)
        p = permutation_p(values)
        pos = sum(1 for v in values if v > 0)
        neg = sum(1 for v in values if v < 0)
        rows.append([label, mean, low, high, p, len(values), pos, neg])

    survivors = benjamini_hochberg([row[4] for row in rows])
    print(f"\n{title}")
    print(f"  {len(rows)} comparisons in this family; BH-FDR applied across all of them at q<{Q}")
    print(f"  {'comparison':<34}{'diff':>8}{'95% CI':>20}{'p':>9}{'n':>5}  {'+/-':>7}  verdict")
    significant = 0
    for row, survives in zip(rows, survivors):
        label, mean, low, high, p, n, pos, neg = row
        if survives:
            significant += 1
        print(f"  {label:<34}{mean:+8.3f}  [{low:+.3f},{high:+.3f}]{p:>9.4f}{n:>5}"
              f"  {pos:>3}+/{neg:<3}  {stars(p, survives)}")
    print(f"  -> {significant} of {len(rows)} survive BH at q<{Q}")
    return significant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    for model in args.models:
        rows = [row for row in read_csv(pilot_csv(model)) if row["error"] == ""]
        means = cell_means(rows, lambda row: as_float(row.get("consensus_credence")))
        cells = [c for c in CELL_ORDER if any(key[2] == c for key in means)]

        print("=" * 100)
        print(f"{model}   ({len(rows)} answered rows)")
        print("=" * 100)

        # 1. means
        print("\nMEAN CREDENCE PER CONDITION (items averaged within cell, then pooled)")
        print(f"  {'condition':<13}{'conv':>9}{'n':>5}{'instr':>10}{'n':>5}{'instr - conv':>15}")
        for cell in cells:
            line = f"  {cell:<13}"
            per_fmt = {}
            for fmt in FORMATS:
                vals = [v for key, v in means.items() if key[1] == fmt and key[2] == cell]
                per_fmt[fmt] = vals
                line += f"{statistics.fmean(vals):>9.3f}{len(vals):>5}" if vals else f"{'-':>9}{0:>5}"
            paired = paired_formats(means, "instr", cell, "conv", cell)
            line += f"{statistics.fmean([v for _, v in paired]):>15.3f}" if paired else f"{'-':>15}"
            print(line)

        # 2. format, within condition
        run_family(
            "FORMAT: instructional minus conversational, within each condition",
            [(cell, paired_formats(means, "instr", cell, "conv", cell)) for cell in cells],
        )

        # 3. all pairs within each format
        for fmt in FORMATS:
            run_family(
                f"PAIRWISE: all condition pairs, {fmt} (left minus right)",
                [(f"{a} - {b}", paired_formats(means, fmt, a, fmt, b))
                 for a, b in itertools.combinations(cells, 2)],
            )
        print()


if __name__ == "__main__":
    main()
