import csv
import sys

import pytest

from presup.design import cell_means, paired
from presup.paths import pilot_csv, read_csv
from presup.stats import as_float, bootstrap

csv.field_size_limit(sys.maxsize)


def contrast(model, fmt, left, right):
    rows = [r for r in read_csv(pilot_csv(model)) if r["error"] == ""]
    means = cell_means(rows, lambda r: as_float(r["consensus_credence"]))
    values = [d for _, d in paired(means, fmt, left, right)]
    return round(sum(values) / len(values), 3), tuple(round(x, 3) for x in bootstrap(values))


# the headline numbers in the README
@pytest.mark.parametrize("model, fmt, left, right, mean, ci", [
    ("anthropic/claude-sonnet-4-6", "conv", "NAI_PRESUP", "NAI_COMMIT", -0.016, (-0.040, 0.010)),
    ("anthropic/claude-sonnet-4-6", "instr", "NAI_PRESUP", "NAI_COMMIT", 0.044, (-0.014, 0.117)),
    ("google/gemini-3.7-flash", "conv", "NAI_PRESUP", "NAI_COMMIT", -0.009, (-0.046, 0.024)),
    ("google/gemini-3.7-flash", "instr", "NAI_PRESUP", "NAI_COMMIT", 0.006, (-0.020, 0.039)),
    ("anthropic/claude-sonnet-4-6", "conv", "AI_HEDGE", "AI_PRESUP", 0.040, (0.017, 0.067)),
    ("google/gemini-3.7-flash", "conv", "AI_HEDGE", "AI_PRESUP", 0.038, (0.019, 0.059)),
])
def test_headline_contrasts(model, fmt, left, right, mean, ci):
    assert contrast(model, fmt, left, right) == (mean, ci)
