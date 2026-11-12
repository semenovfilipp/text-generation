import re
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "You are a helpful assistant that writes short, factual summaries of dialogues."

USER_TEMPLATE = (
    "Summarize the following dialogue in one to three sentences. "
    "Write in the third person and refer to the speakers as #Person1# and #Person2#. "
    "Output only the summary.\n\n"
    "Dialogue:\n{dialogue}"
)

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


def find_csv_dir(raw_dir, kaggle_dataset=None):
    raw_dir = Path(raw_dir)
    hits = sorted(raw_dir.rglob("train.csv")) if raw_dir.exists() else []
    if not hits and kaggle_dataset:
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(kaggle_dataset))
        hits = sorted(downloaded.rglob("train.csv"))
    if not hits:
        raise FileNotFoundError(f"train.csv not found under {raw_dir}")
    return hits[0].parent


def load_split(csv_dir, split):
    return pd.read_csv(Path(csv_dir) / SPLIT_FILES[split])


def group_references(df):
    df = df.copy()
    df["dialogue_id"] = df["id"].str.replace(r"_\d+$", "", regex=True)
    grouped = (
        df.groupby("dialogue_id", sort=False)
        .agg(dialogue=("dialogue", "first"), topic=("topic", "first"), references=("summary", list))
        .reset_index()
        .rename(columns={"dialogue_id": "id"})
    )
    return grouped


def clean_dialogue(text):
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"[ \t]+", " ", text)


def build_messages(dialogue, summary=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary})
    return messages


def postprocess_summary(text):
    text = text.strip()
    text = re.sub(r"^(summary|here is the summary)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())
