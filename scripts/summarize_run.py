"""Summarize saved training logs and judge scores, without loading an LLM."""

import itertools
import json
from pathlib import Path

import click
import numpy as np

from dialogsum.utils import load_config, read_jsonl, write_json


@click.command()
@click.option("--judge-config", default="configs/judge.yaml", show_default=True)
@click.option("--output", default="results/run_summary.json", show_default=True)
def main(judge_config, output):
    cfg = load_config(judge_config)
    history_path = Path("results/training/log_history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train = [r for r in history if "loss" in r]
    validation = [r for r in history if "eval_loss" in r]
    summary = {
        "training": {
            "first_logged_loss": train[0]["loss"],
            "last_logged_loss": train[-1]["loss"],
            "first_eval_loss": validation[0]["eval_loss"],
            "last_eval_loss": validation[-1]["eval_loss"],
            "best_eval_loss": min(r["eval_loss"] for r in validation),
            "final_step": train[-1]["step"],
        },
        "judge_model": cfg["model"],
        "judge_seed": cfg["seed"],
    }
    path = Path(cfg["output_dir"]) / "judge_raw.jsonl"
    records = read_jsonl(path)
    if len(records) != cfg["n_samples"] or len({r["id"] for r in records}) != len(records):
        raise click.ClickException("Judge run is incomplete; do not publish partial comparisons")
    systems = list(cfg["systems"])
    if any(set(r["scores"]) != set(systems) for r in records):
        raise click.ClickException("Judge records contain different system sets")
    summary["judge_n"] = len(records)
    summary["judge_total_tokens"] = sum(r.get("total_tokens", 0) for r in records)
    comparisons = []
    rng = np.random.default_rng(cfg["seed"])
    for a, b in itertools.combinations(systems, 2):
        delta = np.array([r["scores"][a]["overall"] - r["scores"][b]["overall"] for r in records])
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        comparisons.append({
            "a": a, "b": b,
            "a_wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "b_wins": int((delta < 0).sum()),
            "mean_overall_a_minus_b": round(float(delta.mean()), 4),
            "paired_bootstrap_95_ci": [round(float(v), 4) for v in np.quantile(boot, [0.025, 0.975])],
        })
    summary["pairwise_overall"] = comparisons
    write_json(summary, output)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
