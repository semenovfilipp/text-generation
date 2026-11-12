import re
import time

import click

from dialogsum.utils import load_config, read_jsonl, write_json, write_jsonl

TURN_RE = re.compile(r"^(#Person\d+#):\s*(.*)$")
SENT_RE = re.compile(r"(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bMrs\.)(?<!\bJr\.)(?<=[.!?])\s+")


def sentences(dialogue):
    result = []
    for line in dialogue.split("\n"):
        match = TURN_RE.match(line.strip())
        speaker, text = (match.group(1), match.group(2)) if match else ("", line.strip())
        for sent in SENT_RE.split(text):
            if sent.strip():
                result.append(f"{speaker}: {sent.strip()}" if speaker else sent.strip())
    return result


def first_last(dialogue):
    sents = sentences(dialogue)
    if len(sents) <= 1:
        return " ".join(sents)
    return f"{sents[0]} {sents[-1]}"


@click.command()
@click.option("--config", default="configs/data.yaml", show_default=True)
@click.option("--output", default="outputs/predictions/lead_first_last.jsonl", show_default=True)
def main(config, output):
    cfg = load_config(config)
    test = read_jsonl(f"{cfg['processed_dir']}/test.jsonl")
    start = time.perf_counter()
    rows = [{"id": r["id"], "prediction": first_last(r["dialogue"])} for r in test]
    elapsed = time.perf_counter() - start
    write_jsonl(rows, output)
    write_json(
        {"system": "lead_first_last", "n": len(rows), "seconds": elapsed, "samples_per_sec": len(rows) / elapsed},
        output.replace(".jsonl", "_meta.json"),
    )
    click.echo(f"saved {len(rows)} predictions -> {output}")


if __name__ == "__main__":
    main()
