"""Validate and import an extracted Colab archive without importing model weights."""

import hashlib
import json
import shutil
from pathlib import Path

import click

from dialogsum.utils import ensure_dir, read_jsonl, write_json

SYSTEMS = ["lead_first_last", "qwen2.5-0.5b_zeroshot", "qwen2.5-0.5b_lora"]


@click.command()
@click.option("--source", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(source):
    """Import results from SOURCE; raw data and predictions remain gitignored."""
    test = read_jsonl(source / "test.jsonl")
    ids = {r["id"] for r in test}
    if len(test) != 500 or len(ids) != 500 or any(len(r["references"]) != 3 for r in test):
        raise click.ClickException("Expected 500 unique test dialogues with three references")
    required = [source / name for name in ["metrics.csv", "environment.txt", "run_manifest.json", "manual_review.json"]]
    required += list((source / "run_configs").glob("*.yaml"))
    training_source = source / "models/qwen2.5-0.5b-lora"
    required += [training_source / name for name in ["log_history.json", "loss.png", "train_summary.json"]]
    for system in SYSTEMS:
        path = source / "predictions" / f"{system}.jsonl"
        required += [path, path.with_name(f"{system}_meta.json")]
        if not path.is_file():
            raise click.ClickException(f"Missing {path}")
        rows = read_jsonl(path)
        if len(rows) != 500 or {r["id"] for r in rows} != ids:
            raise click.ClickException(f"Incomplete or duplicate predictions: {path}")
        if any(not isinstance(r["prediction"], str) or not r["prediction"].strip() for r in rows):
            raise click.ClickException(f"Empty predictions: {path}")
    if len(list((source / "run_configs").glob("*.yaml"))) != 5:
        raise click.ClickException("Expected five run configuration snapshots")
    for path in required:
        if not path.is_file():
            raise click.ClickException(f"Missing {path}")
    ensure_dir("data/processed")
    ensure_dir("outputs/predictions")
    ensure_dir("results/training")
    ensure_dir("results/run_configs")
    shutil.copyfile(source / "test.jsonl", "data/processed/test.jsonl")
    for path in (source / "predictions").glob("*"):
        if path.suffix in (".json", ".jsonl"):
            shutil.copyfile(path, Path("outputs/predictions") / path.name)
    for path in required:
        if path.parent == training_source:
            target = Path("results/training") / path.name
        elif path.parent == source / "run_configs":
            target = Path("results/run_configs") / path.name
        elif path.parent == source and path.name != "manual_review.json":
            target = Path("results") / path.name
        else:
            continue
        shutil.copyfile(path, target)
    manifest = {
        "test_dialogues": len(test),
        "references_per_dialogue": 3,
        "files_sha256": {
            str(path.relative_to(source)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source / "test.jsonl", *required]
        },
        "source_run": json.loads((source / "run_manifest.json").read_text()),
    }
    write_json(manifest, "results/artifact_manifest.json")
    click.echo("Imported complete Colab results. Data and predictions remain outside Git.")


if __name__ == "__main__":
    main()
