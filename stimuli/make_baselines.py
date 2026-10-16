"""Baseline credence for every Pander Score proposition, used to pick and band the pilot items.

For each of three published models: conversational prompts only, the benchmark's exclusions,
then a per-proposition fit of logit(credence) on logit(author valence). The baseline is the
fitted credence at author valence 0.5, i.e. from_logit(intercept).

Downloads the released Pander Score data from Hugging Face (no model calls) and writes
stimuli/baselines.csv.

    uv run python stimuli/make_baselines.py
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from scipy.stats import linregress

from pander_score.core.metrics import (
    compute_pander_score,
    from_logit,
    to_logit,
    with_pander_score_exclusions,
)

OUTPUT = Path(__file__).resolve().parent / "baselines.csv"
DATASET_REVISION = "233e697f341d63d52d406dd0c5f8b838c4bee671"
MIN_PROMPTS = 4  # the benchmark's per-proposition minimum

# model -> (column name, published conversational score, to check the data loaded right)
MODELS = {
    "openai/gpt-5.4-mini-2026-03-17": ("gpt-5.4-mini", 0.1643480569098327),
    "google/gemini-3.7-flash": ("gemini-3.7-flash", 0.15684783476894384),
    "anthropic/claude-sonnet-4-6": ("claude-sonnet-4-6", -0.039362697054534095),
}

BANDS = [(0.1, "floor"), (0.4, "doubted"), (0.6, "open"), (0.9, "believed")]  # else ceiling


def band(value: float | None) -> str | None:
    if value is None:
        return None
    for upper, label in BANDS:
        if value < upper or (label == "believed" and value <= upper):
            return label
    return "ceiling"


def download(name: str) -> Path:
    return Path(hf_hub_download("sophronresearch/pander-score", f"v1/benchmark/{name}",
                                repo_type="dataset", revision=DATASET_REVISION))


def load_rows(model_id: str, attrs: pl.DataFrame) -> pl.DataFrame:
    # same join as pander_score.data
    model = pl.read_parquet(download("per_model/" + model_id.replace("/", "-") + ".parquet"))
    if "proposition_id" in model.columns and "proposition_id" in attrs.columns:
        attrs = attrs.drop("proposition_id")
    joined = model.join(attrs, on="sample_id", how="left")
    assert joined.height == model.height
    return joined


def baseline(group: pl.DataFrame) -> float | None:
    data = group.select(["consensus_author_valence", "consensus_credence"]).drop_nulls()
    if data.height < MIN_PROMPTS:
        return None
    x = data["consensus_author_valence"].to_numpy().astype(float)
    y = data["consensus_credence"].to_numpy().astype(float)
    if np.std(x) < 1e-12:
        return None
    fit = linregress(to_logit(x), to_logit(y))
    if not math.isfinite(fit.slope):
        return None
    return float(from_logit(np.array(fit.intercept)))


def main() -> None:
    propositions = pl.read_csv(download("propositions.csv"))
    assert propositions.height == 349
    attrs = pl.read_parquet(download("prompt_attributes.parquet"))
    table = propositions.select(["id", "domain", "proposition"])

    columns = []
    for model_id, (short, published) in MODELS.items():
        conversational = load_rows(model_id, attrs).filter(~pl.col("is_artifact"))
        score = compute_pander_score(conversational).score
        if abs(score - published) > 1e-9:
            raise SystemExit(f"{model_id}: score {score} does not match the published {published}")
        kept = with_pander_score_exclusions(conversational)
        values = {prop: baseline(kept.filter(pl.col("proposition_id") == prop)) for prop in table["id"]}
        column = f"baseline_{short}"
        columns.append(column)
        table = table.with_columns(pl.col("id").replace_strict(values, return_dtype=pl.Float64).alias(column))
        skipped = sum(v is None for v in values.values())
        print(f"{short:18} score {100 * score:+.1f} (matches published), "
              f"{skipped}/349 propositions with <{MIN_PROMPTS} prompts left")

    # average only where all three models have a baseline
    table = table.with_columns(
        pl.when(pl.all_horizontal([pl.col(c).is_not_null() for c in columns]))
        .then(pl.mean_horizontal(columns))
        .otherwise(None)
        .alias("baseline_avg"),
        (pl.max_horizontal(columns) - pl.min_horizontal(columns)).alias("model_spread"),
    )
    table = table.with_columns(pl.col("baseline_avg").map_elements(band, return_dtype=pl.String).alias("band"))
    table.write_csv(OUTPUT)
    print(table.group_by("band").len().sort("band"))
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
