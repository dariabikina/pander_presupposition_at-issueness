"""Condition means, planned contrasts, challenge rates and stated credence, per model.

Reads results/pilot_<model>.csv. Rows with a retryable error are dropped and empty responses
are counted but not analysed. The item is the unit of analysis.

The effect is reported as credence differences between cells rather than as the benchmark's
slope on author valence: the claim-bearing cells all sit at valence 0.93-0.98, and the valence
judges read a presupposing frame as a belief cue, so a slope would partly regress the
manipulation on itself.

    uv run python analysis/analyse.py --models google/gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from presup import challenge as chal
from presup.design import (CELL_ORDER, ENCODINGS, FORMATS, cell_means, paired,
                           paired_combination)
from presup.paths import MODELS, pilot_csv, read_csv
from presup.stats import as_float, bootstrap, bootstrap_statistic, ols_slope, spearman

# (label, left cell, right cell), every contrast is left minus right
CONTRASTS = (
    ("PRIMARY: factivity at matched commitment", "NAI_PRESUP", "NAI_COMMIT"),
    ("factivity vs hedged assertion", "NAI_PRESUP", "NAI_HEDGE"),
    ("commitment alone, both non-factive", "NAI_COMMIT", "NAI_HEDGE"),
    ("presupposed - bare assertion", "NAI_PRESUP", "NAI_ASSERT"),
)

# split point on the item's baseline credence for the moderator, fixed in advance
MEDIAN_SPLIT = 0.35


def claim_cells(rows: list[dict]) -> tuple[str, ...]:
    """The claim-bearing conditions present in these rows, in CELL_ORDER."""
    present = {row["condition"] for row in rows}
    unknown = present - set(CELL_ORDER)
    if unknown:
        raise SystemExit(
            f"unknown condition(s) in the results: {sorted(unknown)}\n"
            f"  known: {list(CELL_ORDER)}\n"
            "  add them to CELL_ORDER (and to the contrast list)"
        )
    return tuple(cell for cell in CELL_ORDER if cell in present and cell != "NO_CLAIM")


def contrast(label: str, differences: list[tuple[str, float]], width: int = 46) -> None:
    if not differences:
        print(f"  {label:{width}} no items")
        return
    values = [d for _, d in differences]
    low, high = bootstrap(values)
    positive = sum(1 for v in values if v > 1e-12)
    negative = sum(1 for v in values if v < -1e-12)
    print(f"  {label:{width}} {statistics.fmean(values):+.3f}  [{low:+.3f}, {high:+.3f}]  "
          f"n={len(values)} items ({positive}+/{negative}-)")


def moderator(label: str, differences: list[tuple[str, float]],
              baselines: dict[str, float]) -> None:
    """Per-item effect against the item's baseline credence: slope and rho, then the split."""
    usable = [(baselines[item], value) for item, value in differences if item in baselines]
    missing = len(differences) - len(usable)
    print(f"  {label}")
    if len(usable) < 3:
        print(f"    only {len(usable)} item(s) with both an effect and a baseline; skipped")
        return
    xs = [x for x, _ in usable]
    ys = [y for _, y in usable]
    if missing:
        print(f"    note: {missing} item(s) dropped for a missing baseline_avg")

    slope = ols_slope(xs, ys)
    slope_low, slope_high = bootstrap_statistic(usable, ols_slope)
    rho = spearman(xs, ys)
    rho_low, rho_high = bootstrap_statistic(usable, spearman)
    print(f"    continuous  slope {slope:+.3f} [{slope_low:+.3f}, {slope_high:+.3f}] "
          f"per unit of baseline credence   (n={len(usable)} items)")
    print(f"    continuous  Spearman rho {rho:+.3f} [{rho_low:+.3f}, {rho_high:+.3f}]  "
          f"(n={len(usable)} items)")
    print(f"    baseline_avg spans {min(xs):.3f} to {max(xs):.3f}, "
          f"median {statistics.median(xs):.3f} (n={len(usable)} items)")

    low_band = [y for x, y in usable if x < MEDIAN_SPLIT]
    high_band = [y for x, y in usable if x >= MEDIAN_SPLIT]
    print(f"    split at baseline {MEDIAN_SPLIT} (fixed in advance, not fitted here)")
    for name, band in ((f"baseline < {MEDIAN_SPLIT}", low_band),
                       (f"baseline >= {MEDIAN_SPLIT}", high_band)):
        if band:
            band_low, band_high = bootstrap(band)
            print(f"      {name:20} {statistics.fmean(band):+.3f} "
                  f"[{band_low:+.3f}, {band_high:+.3f}]  n={len(band)} items")
        else:
            print(f"      {name:20} no items")
    if low_band and high_band:
        # unpaired: the two bands are different items
        gap = statistics.fmean(high_band) - statistics.fmean(low_band)
        print(f"      high minus low       {gap:+.3f}  "
              f"(n={len(high_band)} vs {len(low_band)} items, unpaired)")


def report_credence(rows: list[dict], cells: tuple[str, ...], baselines: dict[str, float]) -> None:
    means = cell_means(rows, lambda row: as_float(row["consensus_credence"]))

    print("\n--- mean credence per condition (items averaged, then pooled)")
    for fmt in FORMATS:
        for condition in ("NO_CLAIM", *cells):
            values = [v for key, v in means.items() if key[1] == fmt and key[2] == condition]
            if values:
                print(f"  {fmt:5} {condition:11} {statistics.fmean(values):.3f}  "
                      f"(n={len(values)} items)")

    print("\n--- CONTRASTS, paired by item, bootstrap over items")
    for label, left, right in CONTRASTS:
        if left not in cells or right not in cells:
            continue
        print(f"\n  {label}: {left} - {right}")
        for fmt in FORMATS:
            contrast(f"  {fmt:5} not-at-issue", paired(means, fmt, left, right))
            # the same contrast at the at-issue level
            ai_left, ai_right = left.replace("NAI_", "AI_"), right.replace("NAI_", "AI_")
            if ai_left in cells and ai_right in cells:
                contrast(f"  {fmt:5} at-issue    ", paired(means, fmt, ai_left, ai_right))

    print("\n--- 2 x 4 MAIN EFFECTS (collapsing the other factor, paired by item)")
    print("  Encoding, each framing minus the bare assertion, averaged over at-issueness:")
    for fmt in FORMATS:
        for encoding in ENCODINGS[1:]:
            weights = {f"NAI_{encoding}": 0.5, f"AI_{encoding}": 0.5,
                       "NAI_ASSERT": -0.5, "AI_ASSERT": -0.5}
            if not set(weights) <= set(cells):
                continue
            contrast(f"  {fmt:5} {encoding} - ASSERT", paired_combination(means, fmt, weights))
    print("  At-issueness, not-at-issue minus at-issue, averaged over the four framings:")
    weights = {f"NAI_{encoding}": 0.25 for encoding in ENCODINGS}
    weights |= {f"AI_{encoding}": -0.25 for encoding in ENCODINGS}
    if set(weights) <= set(cells):
        for fmt in FORMATS:
            contrast(f"  {fmt:5} NAI - AI", paired_combination(means, fmt, weights))

    print("\n--- ENCODING x AT-ISSUENESS INTERACTION")
    print("  (NAI_x - NAI_ref) - (AI_x - AI_ref); zero = same encoding effect at both levels")
    for fmt in FORMATS:
        for left, right in (("PRESUP", "COMMIT"), ("PRESUP", "ASSERT"), ("COMMIT", "HEDGE")):
            weights = {f"NAI_{left}": 1.0, f"NAI_{right}": -1.0,
                       f"AI_{left}": -1.0, f"AI_{right}": 1.0}
            if not set(weights) <= set(cells):
                continue
            contrast(f"  {fmt:5} ({left} - {right}) x (NAI - AI)",
                     paired_combination(means, fmt, weights))

    print("\n--- deference relative to the no-claim anchor (where the anchor is informative)")
    for fmt in FORMATS:
        for condition in cells:
            contrast(f"{fmt:5} {condition} - NO_CLAIM", paired(means, fmt, condition, "NO_CLAIM"))

    print("\n--- MODERATOR: is the effect larger on claims the model already half believes?")
    print("  The per-item effect regressed on the item's released baseline credence.")
    for label, left, right in CONTRASTS:
        if left not in cells or right not in cells:
            continue
        for fmt in FORMATS:
            differences = paired(means, fmt, left, right)
            if differences:
                moderator(f"{fmt:5} {left} - {right}  ({label})", differences, baselines)


def report_challenge(rows: list[dict], cells: tuple[str, ...],
                     baselines: dict[str, float]) -> None:
    print("\n--- CHALLENGE: did the response push back on the claim")
    print(f"  source: {chal.source_note(rows)}")
    print("  the user states no claim in NO_CLAIM, so its rate is the judge's false-positive "
          "baseline")

    variants = chal.available_variants(rows)
    if not variants:
        print("  no challenge columns in this file; nothing to report")
        return

    for variant in variants:
        print(f"\n  instrument: {variant}"
              + ("  (PRIMARY - the judge cannot see the framing)"
                 if variant == chal.PRIMARY_VARIANT else "  (framing visible to the judge)"))
        print(f"    {chal.format_agreement(chal.agreement(rows, variant))}")
        pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            value = chal.score(row, variant)
            if value is not None:
                pooled[(row["format"], row["condition"])].append(value)
        for fmt in FORMATS:
            baseline_values = pooled.get((fmt, "NO_CLAIM"))
            baseline = statistics.fmean(baseline_values) if baseline_values else None
            for condition in ("NO_CLAIM", *cells):
                values = pooled.get((fmt, condition))
                if not values:
                    continue
                rate = statistics.fmean(values)
                note = ("<- false-positive baseline" if condition == "NO_CLAIM"
                        else chal.baseline_corrected(rate, baseline))
                print(f"    {fmt:5} {condition:11} {rate:5.0%} of {len(values):3} responses"
                      f"   {note}")

    if len(variants) < 2:
        print("\n  frame contamination: needs both instruments; this file has only "
              f"{list(variants)}")
        return

    print("\n--- FRAME CONTAMINATION: blinded minus unblinded, the same response twice")
    print("  a gap means the user's framing moved the judge, not the response")
    print(f"  {chal.format_agreement(chal.frame_contamination(rows))}")
    blind = cell_means(rows, lambda row: chal.score(row, "blind"))
    open_ = cell_means(rows, lambda row: chal.score(row, "open"))
    shared = sorted(set(blind) & set(open_))
    print("\n  per cell, paired by item (blinded minus unblinded):")
    for fmt in FORMATS:
        for condition in ("NO_CLAIM", *cells):
            differences = [(key[0], blind[key] - open_[key]) for key in shared
                           if key[1] == fmt and key[2] == condition]
            contrast(f"  {fmt:5} {condition:11}", differences, width=26)

    print("\n  the contrasts under each instrument:")
    for label, left, right in CONTRASTS:
        if left not in cells or right not in cells:
            continue
        for fmt in FORMATS:
            for name, means in (("blind", blind), ("open ", open_)):
                contrast(f"  {fmt:5} {name} {left} - {right}", paired(means, fmt, left, right))

    print("\n--- MODERATOR on the blinded challenge measure")
    for label, left, right in CONTRASTS[:1]:
        for fmt in FORMATS:
            differences = paired(blind, fmt, left, right)
            if differences:
                moderator(f"{fmt:5} blind {left} - {right}", differences, baselines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    for model in args.models:
        analyse(model)


def analyse(model: str) -> None:
    path = pilot_csv(model)
    raw = read_csv(path)
    print(f"reading {path.name}")
    stimuli_tags = {row.get("stimuli") for row in raw if row.get("stimuli")}
    if stimuli_tags:
        print(f"  provenance: rows came from {sorted(stimuli_tags)}")
    claim_fields = {row.get("claim_field") for row in raw if row.get("claim_field")}
    if claim_fields:
        print(f"  the challenge judge was asked about: {sorted(claim_fields)}")

    answered = [r for r in raw if r["error"] == ""]
    empty = [r for r in raw if r["error"] == "empty_response"]
    failed = [r for r in raw if r["error"] not in ("", "empty_response")]
    print(f"rows {len(raw)}: answered {len(answered)}, empty {len(empty)}, "
          f"retryable errors {len(failed)}")
    if not answered:
        raise SystemExit("no answered rows in this file")

    cells = claim_cells(answered)
    print(f"conditions present: {['NO_CLAIM', *cells]}")
    baselines = {row["item_id"]: value for row in answered
                 if (value := as_float(row.get("baseline_avg"))) is not None}
    print(f"baseline_avg available for {len(baselines)} of "
          f"{len({r['item_id'] for r in answered})} items")

    print("\n--- informativeness (credence recoverable from the response)")
    by_condition: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in answered:
        by_condition[(row["condition"], row["format"])].append(
            1 if as_float(row["consensus_credence"]) is not None else 0)
    for (condition, fmt), flags in sorted(by_condition.items()):
        print(f"  {condition:11} {fmt:5} {statistics.fmean(flags):5.0%} of {len(flags)} responses")

    informative = [r for r in answered if as_float(r["consensus_credence"]) is not None]
    report_credence(informative, cells, baselines)
    report_challenge(answered, cells, baselines)

    print("\n--- COMPLIANCE TEST: judged credence minus stated credence")
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in answered:
        judged, stated = as_float(row["consensus_credence"]), as_float(row["stated_credence"])
        if judged is not None and stated is not None:
            gaps[(row["condition"], row["format"])].append(judged - stated)
    for (condition, fmt), values in sorted(gaps.items()):
        low, high = bootstrap(values)
        print(f"  {condition:11} {fmt:5} {statistics.fmean(values):+.3f}  "
              f"[{low:+.3f}, {high:+.3f}]  n={len(values)} responses")


if __name__ == "__main__":
    main()
