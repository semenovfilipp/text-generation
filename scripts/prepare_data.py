import click

from dialogsum.data import clean_dialogue, find_csv_dir, group_references, load_split
from dialogsum.utils import load_config, set_seed, write_jsonl


def to_rows(df, with_references):
    rows = []
    for rec in df.to_dict("records"):
        row = {"id": rec["id"], "topic": rec["topic"], "dialogue": clean_dialogue(rec["dialogue"])}
        if with_references:
            row["references"] = [r.strip() for r in rec["references"]]
        else:
            row["summary"] = rec["summary"].strip()
        rows.append(row)
    return rows


@click.command()
@click.option("--config", default="configs/data.yaml", show_default=True)
def main(config):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    csv_dir = find_csv_dir(cfg["raw_dir"], cfg.get("kaggle_dataset"))
    click.echo(f"raw csv dir: {csv_dir}")

    train = load_split(csv_dir, "train").sample(n=cfg["train_size"], random_state=cfg["seed"])
    val = load_split(csv_dir, "validation").sample(n=cfg["val_size"], random_state=cfg["seed"])
    test = group_references(load_split(csv_dir, "test"))

    out = cfg["processed_dir"]
    write_jsonl(to_rows(train, False), f"{out}/train.jsonl")
    write_jsonl(to_rows(val, False), f"{out}/val.jsonl")
    write_jsonl(to_rows(test, True), f"{out}/test.jsonl")
    click.echo(f"train={len(train)} val={len(val)} test_dialogues={len(test)} -> {out}")


if __name__ == "__main__":
    main()
