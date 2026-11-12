import time

import click

from dialogsum.data import build_messages, postprocess_summary
from dialogsum.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def build_prompts(tokenizer, rows):
    return [
        tokenizer.apply_chat_template(build_messages(r["dialogue"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]


def generate_vllm(cfg, prompts):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model_path"],
        dtype=cfg.get("dtype", "auto"),
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        seed=cfg["seed"],
    )
    params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    return [o.outputs[0].text for o in outputs], elapsed


def generate_hf(cfg, prompts, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.float32)
    model.eval()
    tokenizer.padding_side = "left"
    texts = []
    start = time.perf_counter()
    for i in range(0, len(prompts), cfg.get("hf_batch_size", 4)):
        batch = tokenizer(prompts[i : i + cfg.get("hf_batch_size", 4)], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
        texts += tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    return texts, time.perf_counter() - start


def gpu_name():
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@click.command()
@click.option("--config", default="configs/infer_zeroshot.yaml", show_default=True)
@click.option("--model-path", default=None, help="Override model_path from config")
@click.option("--backend", type=click.Choice(["vllm", "hf"]), default=None, help="Override backend from config")
@click.option("--debug", is_flag=True, help="Run on a few examples")
def main(config, model_path, backend, debug):
    cfg = load_config(config)
    if model_path:
        cfg["model_path"] = model_path
    if backend:
        cfg["backend"] = backend
    set_seed(cfg["seed"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    rows = read_jsonl(cfg["test_path"])
    if debug:
        rows = rows[: cfg.get("debug_size", 8)]
    prompts = build_prompts(tokenizer, rows)

    if cfg["backend"] == "vllm":
        texts, elapsed = generate_vllm(cfg, prompts)
    else:
        texts, elapsed = generate_hf(cfg, prompts, tokenizer)

    output = cfg["output_path"].replace(".jsonl", "_debug.jsonl") if debug else cfg["output_path"]
    write_jsonl(
        [{"id": r["id"], "prediction": postprocess_summary(t), "raw": t} for r, t in zip(rows, texts)], output
    )
    meta = {
        "system": cfg["name"],
        "model_path": cfg["model_path"],
        "backend": cfg["backend"],
        "n": len(rows),
        "seconds": elapsed,
        "samples_per_sec": len(rows) / elapsed,
        "gpu": gpu_name(),
    }
    write_json(meta, output.replace(".jsonl", "_meta.json"))
    click.echo(meta)


if __name__ == "__main__":
    main()
