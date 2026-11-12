import json
import os
import random
import time
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

from dialogsum.utils import ensure_dir, load_config, read_jsonl

CRITERIA = ["faithfulness", "coverage", "conciseness", "fluency", "overall"]

JUDGE_PROMPT = """You are an expert evaluator of dialogue summaries.
Read the dialogue and rate each candidate summary independently on a 1-5 integer scale:
- faithfulness: no hallucinated or contradicted facts (5 = fully supported by the dialogue)
- coverage: captures the main point and key details of the dialogue
- conciseness: short and to the point, no filler, no copying of whole turns
- fluency: grammatical, coherent third-person summary
- overall: overall quality as a summary of this dialogue

Dialogue:
{dialogue}

Candidates:
{candidates}

Answer with JSON only, keyed by candidate label, for example:
{{"A": {{"faithfulness": 5, "coverage": 4, "conciseness": 4, "fluency": 5, "overall": 4}}, "B": {{...}}}}"""


def api_key():
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        raise click.ClickException("Set GROQ_API_KEY in .env")
    return key


def call_judge(client, cfg, prompt):
    for attempt in range(cfg["max_retries"]):
        try:
            kwargs = {"reasoning_effort": cfg["reasoning_effort"]} if cfg.get("reasoning_effort") else {}
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return json.loads(resp.choices[0].message.content), resp.usage.total_tokens
        except Exception as exc:
            wait = min(60, 2 ** attempt * 3)
            click.echo(f"retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:150]} (sleep {wait}s)")
            time.sleep(wait)
    return None, 0


@click.command()
@click.option("--config", default="configs/judge.yaml", show_default=True)
@click.option("--debug", is_flag=True, help="Judge only a few dialogues")
def main(config, debug):
    from openai import OpenAI

    cfg = load_config(config)
    client = OpenAI(api_key=api_key(), base_url=cfg["base_url"])
    test = {r["id"]: r for r in read_jsonl(cfg["test_path"])}
    systems = {}
    for name, path in cfg["systems"].items():
        if Path(path).exists():
            systems[name] = {p["id"]: p["prediction"] for p in read_jsonl(path)}
        else:
            click.echo(f"skip {name}: {path} not found")
    common = sorted(set(test).intersection(*[set(v) for v in systems.values()]), key=lambda x: int(x.split("_")[-1]))
    rng = random.Random(cfg["seed"])
    n = cfg["debug_size"] if debug else cfg["n_samples"]
    sample = rng.sample(common, min(n, len(common)))

    raw_path = Path(cfg["output_dir"]) / ("judge_raw_debug.jsonl" if debug else "judge_raw.jsonl")
    ensure_dir(raw_path.parent)
    done = {json.loads(line)["id"] for line in raw_path.open(encoding="utf-8")} if raw_path.exists() else set()
    interval = 60.0 / cfg["requests_per_minute"]
    tokens = 0
    with raw_path.open("a", encoding="utf-8") as f:
        for i, dialogue_id in enumerate(sample):
            if dialogue_id in done:
                continue
            names = list(systems)
            rng.shuffle(names)
            labels = {chr(65 + j): name for j, name in enumerate(names)}
            candidates = "\n".join(f"{label}: {systems[name][dialogue_id]}" for label, name in labels.items())
            prompt = JUDGE_PROMPT.format(dialogue=test[dialogue_id]["dialogue"], candidates=candidates)
            start = time.perf_counter()
            scores, used = call_judge(client, cfg, prompt)
            tokens += used
            if scores:
                record = {"id": dialogue_id, "scores": {labels[k]: v for k, v in scores.items() if k in labels}}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            click.echo(f"[{i + 1}/{len(sample)}] {dialogue_id} tokens={used}")
            time.sleep(max(0.0, interval - (time.perf_counter() - start)))

    records = read_jsonl(raw_path)
    rows = [{"id": r["id"], "system": s, **{c: v.get(c) for c in CRITERIA}} for r in records for s, v in r["scores"].items()]
    df = pd.DataFrame(rows)
    for c in CRITERIA:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    summary = df.groupby("system")[CRITERIA].mean().round(2)
    best = df.loc[df.groupby("id")["overall"].transform("max") == df["overall"]]
    summary["win_rate"] = (best.groupby("system")["id"].nunique() / df["id"].nunique()).round(2)
    summary["n"] = df.groupby("system")["id"].nunique()
    summary = summary.sort_values("overall", ascending=False)
    out_csv = Path(cfg["output_dir"]) / ("judge_scores_debug.csv" if debug else "judge_scores.csv")
    summary.to_csv(out_csv)
    click.echo(f"tokens used this run: {tokens}")
    click.echo(summary.to_markdown())


if __name__ == "__main__":
    main()
