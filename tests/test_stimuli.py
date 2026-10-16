import csv
import importlib.util
import io
import subprocess
import sys

from presup.blinding import check_blinding
from presup.paths import MODELS, ROOT, STIMULI, pilot_csv, read_csv
from presup.stimuli import load_stimuli


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "stimuli" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stimuli_csv_matches_generator():
    make_stimuli = load_script("make_stimuli")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=make_stimuli.HEADER)
    writer.writeheader()
    writer.writerows(row for item in make_stimuli.ITEMS for row in make_stimuli.build(item))
    assert out.getvalue().encode("utf-8") == STIMULI.read_bytes()


def test_design_size():
    records = load_stimuli()
    assert len(records) == 360
    assert len({r["item_id"] for r in records}) == 20


def test_check_stimuli_passes():
    done = subprocess.run([sys.executable, str(ROOT / "stimuli" / "check_stimuli.py")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
    assert "all invariants pass" in done.stdout


def test_blinding_hides_framing():
    assert check_blinding(load_stimuli()) == []


def test_results_used_these_prompts():
    # the runs were made from stimuli_v4.xlsx; stimuli.csv is an exact export of it
    prompts = {r["sample_id"]: r["prompt_text"] for r in load_stimuli()}
    csv.field_size_limit(sys.maxsize)
    for model in MODELS:
        for row in read_csv(pilot_csv(model)):
            assert row["prompt_text"] == prompts[row["sample_id"]]
