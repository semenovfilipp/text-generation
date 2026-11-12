import os
import time

import click

from dialogsum.utils import ensure_dir, load_config, read_jsonl, set_seed, write_json


def use_wandb(cfg):
    if not cfg["wandb"].get("enabled", True):
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb

        return bool(wandb.api.api_key)
    except Exception:
        return False


def load_unsloth(cfg):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        dtype=None,
    )
    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    return model, tokenizer


def load_peft(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype)
    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_cfg), tokenizer


def to_text_dataset(rows, tokenizer):
    from datasets import Dataset

    from dialogsum.data import build_messages

    texts = [tokenizer.apply_chat_template(build_messages(r["dialogue"], r["summary"]), tokenize=False) for r in rows]
    return Dataset.from_dict({"text": texts})


def to_prompt_completion_dataset(rows):
    from datasets import Dataset

    from dialogsum.data import build_messages

    prompts, completions = [], []
    for r in rows:
        messages = build_messages(r["dialogue"], r["summary"])
        prompts.append(messages[:-1])
        completions.append(messages[-1:])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def plot_losses(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(7, 4))
    if train:
        plt.plot(*zip(*train), label="train loss")
    if evals:
        plt.plot(*zip(*evals), marker="o", label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)


@click.command()
@click.option("--config", default="configs/train_lora.yaml", show_default=True)
@click.option("--backend", type=click.Choice(["unsloth", "peft"]), default=None, help="Override backend from config")
@click.option("--debug", is_flag=True, help="Few steps on a tiny subset")
def main(config, backend, debug):
    cfg = load_config(config)
    if backend:
        cfg["backend"] = backend
    if cfg["backend"] == "unsloth":
        __import__("unsloth")
    set_seed(cfg["seed"])
    tr = cfg["training"]
    train_rows = read_jsonl(cfg["train_path"])
    val_rows = read_jsonl(cfg["val_path"])
    if debug:
        train_rows, val_rows = train_rows[:16], val_rows[:4]
        tr = {**tr, "max_steps": 4, "logging_steps": 1, "eval_steps": 2, "per_device_train_batch_size": 2}
        cfg["output_dir"] = cfg["output_dir"] + "-debug"
        cfg["merged_dir"] = cfg["merged_dir"] + "-debug"

    if cfg["backend"] == "unsloth":
        model, tokenizer = load_unsloth(cfg)
        train_ds, val_ds = to_text_dataset(train_rows, tokenizer), to_text_dataset(val_rows, tokenizer)
    else:
        model, tokenizer = load_peft(cfg)
        train_ds, val_ds = to_prompt_completion_dataset(train_rows), to_prompt_completion_dataset(val_rows)

    import torch
    from trl import SFTConfig, SFTTrainer

    report_to = "wandb" if use_wandb(cfg) and not debug else "none"
    if report_to == "wandb":
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg["wandb"].get("run_name"),
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        num_train_epochs=tr["num_train_epochs"],
        max_steps=tr.get("max_steps", -1),
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_steps=tr["warmup_steps"],
        weight_decay=tr["weight_decay"],
        logging_steps=tr["logging_steps"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        save_strategy="no",
        optim=tr["optim"] if torch.cuda.is_available() else "adamw_torch",
        bf16=bf16,
        fp16=fp16,
        max_length=cfg["max_seq_length"],
        seed=cfg["seed"],
        report_to=report_to,
        **({"dataset_text_field": "text"} if cfg["backend"] == "unsloth" else {}),
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=args)
    if cfg["backend"] == "unsloth":
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"
        )

    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    out = ensure_dir(cfg["output_dir"])
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if cfg["backend"] == "unsloth":
        model.save_pretrained_merged(cfg["merged_dir"], tokenizer, save_method="merged_16bit")
    else:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(cfg["merged_dir"])
        tokenizer.save_pretrained(cfg["merged_dir"])

    history = trainer.state.log_history
    write_json(history, out / "log_history.json")
    plot_losses(history, out / "loss.png")
    summary = {
        "backend": cfg["backend"],
        "train_examples": len(train_rows),
        "train_runtime_sec": elapsed,
        "train_loss": result.training_loss,
        "final_eval_loss": next((h["eval_loss"] for h in reversed(history) if "eval_loss" in h), None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "report_to": report_to,
    }
    write_json(summary, out / "train_summary.json")
    click.echo(summary)


if __name__ == "__main__":
    main()
