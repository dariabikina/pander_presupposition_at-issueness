import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
STIMULI = ROOT / "stimuli" / "stimuli.csv"
TEMPLATES = ROOT / "templates"

MODELS = ("google/gemini-3.7-flash", "anthropic/claude-sonnet-4-6")


def pilot_csv(model: str) -> Path:
    return RESULTS / f"pilot_{model.replace('/', '-')}.csv"


def prompt_side_csv() -> Path:
    return RESULTS / "prompt_side.csv"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_path(out_path: Path) -> Path:
    return out_path.with_suffix(".manifest.json")


def check_manifest(out_path: Path, stimuli: Path, create: bool = True, **conditions) -> dict:
    """Refuse to resume a results file written from different stimuli or settings.

    The manifest is only written when `create` is set, so a dry run leaves results/ alone.
    """
    digest = {"stimuli": stimuli.name,
              "sha256_12": hashlib.sha256(stimuli.read_bytes()).hexdigest()[:12],
              **dict(sorted(conditions.items()))}
    manifest = manifest_path(out_path)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != digest:
            raise SystemExit(f"{out_path.name} was written with {previous}, this run uses {digest}.\n"
                             "Move the old file and its manifest aside to start a new run.")
    elif create:
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(digest, indent=2), encoding="utf-8")
        tmp.replace(manifest)
    return digest
