#!/usr/bin/env python3
"""SFT reasoning on top of the adapter that already knows how to respond.

Methodological correction compared to the first attempt:

  BEFORE: trained reasoning from the BASE model. The model had never
         learned to respond, so comparison with direct SFT was invalid —
         one arm had response ability and the other did not.

  NOW: starts from the move adapter and ADDS reasoning on top.
         Thus the only changing factor is the reasoning format, and comparison
         becomes causal.

Data: traces with candidate exploration, verified by rejection sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_traces(path: str, limit: int | None) -> Dataset:
    messages = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit and len(messages) >= limit:
                break
            d = json.loads(line)
            messages.append([
                {"role": "user", "content": d["prompt"]},
                {"role": "assistant", "content": d["response"]},
            ])
    return Dataset.from_dict({"messages": messages})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--base-adapter", default="out/sft-adapter",
                    help="Move adapter; reasoning is added on top")
    ap.add_argument("--data", default="data/traces.jsonl")
    ap.add_argument("--samples", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=64,
                    help="new LoRA rank when no base adapter is provided")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs")
    ap.add_argument("--checkpointing", type=int, default=1, help="1 enables, 0 disables")
    ap.add_argument("--out", default="out/sft")
    args = ap.parse_args()

    ds = load_traces(args.data, args.samples or None)
    print("traces loaded:", len(ds), flush=True)
    print("example:", ds[0]["messages"][1]["content"][:260].replace("\n", " | "), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")

    if args.base_adapter != "none" and Path(args.base_adapter).exists():
        print("starting from adapter:", args.base_adapter, flush=True)
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    else:
        print("New LoRA r=%d on the base model" % args.lora_r, flush=True)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        logging_steps=25, save_strategy="no", bf16=True, max_length=args.max_length,
        max_steps=args.max_steps,
        completion_only_loss=True, gradient_checkpointing=bool(args.checkpointing),
        report_to=[], seed=args.seed, dataloader_num_workers=2,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                          processing_class=tok)
    print("training reasoning on the adapter...", flush=True)
    trainer.train()
    print("Peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    Path("out").mkdir(exist_ok=True)
    # Name derived from output to avoid overwriting another run's metric
    metrics_path = Path("out") / (Path(args.out).name + "-metrics.json")
    metrics_path.write_text(json.dumps({
        "examples": len(ds), "epochs": args.epochs, "lr": args.lr,
        "adapter_base": args.base_adapter,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": trainer.state.log_history[-4:],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("saved to", args.out, "| metrics at", metrics_path)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
