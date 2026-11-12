import json
from pathlib import Path

import click
import pandas as pd

from dialogsum.metrics import compute_metrics
from dialogsum.utils import ensure_dir, load_config, read_jsonl


@click.command()
@click.option("--config", default="configs/data.yaml", show_default=True)
@click.option("--predictions-dir", default="outputs/predictions", show_default=True)
@click.option("--output", default="results/metrics.csv", show_default=True)
def main(config, predictions_dir, output):
    cfg = load_config(config)
    test = {r["id"]: r for r in read_jsonl(f"{cfg['processed_dir']}/test.jsonl")}
    rows = []
    for path in sorted(Path(predictions_dir).glob("*.jsonl")):
        if "_debug" in path.stem:
            continue
        preds = read_jsonl(path)
        ids = [p["id"] for p in preds]
        if len(ids) != len(set(ids)) or set(ids) != set(test):
            raise click.ClickException(f"{path}: expected every test ID exactly once")
        metrics = compute_metrics([p["prediction"] for p in preds], [test[p["id"]]["references"] for p in preds])
        meta_path = path.with_name(path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows.append(
            {
                "system": path.stem,
                "n": len(preds),
                **{k: round(v, 2) for k, v in metrics.items()},
                "samples_per_sec": round(meta.get("samples_per_sec", float("nan")), 2),
                "gpu": meta.get("gpu", ""),
            }
        )
    if not rows:
        raise click.ClickException("No complete prediction files found")
    table = pd.DataFrame(rows)
    ensure_dir(Path(output).parent)
    table.to_csv(output, index=False)
    click.echo(table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string(index=False))


if __name__ == "__main__":
    main()
