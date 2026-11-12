#!/usr/bin/env python3
"""Compare generalization on Connect Four and GSM8K using a base model or adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.connect4 import legal_columns, render, generate_positions
from chess_rlvr.evaluation import wilson

# Keep measured prompt text intact; LEGACY_FORMAT defaults to the measured language.
LEGACY_FORMAT = True
LEGACY_CONNECT4_PROMPT = "Você está jogando Lig 4. As peças caem até a casa livre mais baixa da coluna. Vence quem alinhar 4 peças na horizontal, vertical ou diagonal.\n\n{board}\n\nÉ a vez de {side}.\nResponda apenas com o número da coluna (1 a 7)."
ENGLISH_CONNECT4_PROMPT = "You are playing Connect Four. Pieces fall to the lowest empty cell in a column. The first player to align 4 pieces horizontally, vertically, or diagonally wins.\n\n{board}\n\nIt is {side}'s turn.\nReply only with the column number (1 to 7)."
LEGACY_GSM8K_SUFFIX = "\n\nResolva passo a passo e termine com a linha 'Resposta: <número>'."
ENGLISH_GSM8K_SUFFIX = "\n\nSolve step by step and finish with the line 'Answer: <number>'."


def connect4_prompt(position: dict) -> str:
    prompt = LEGACY_CONNECT4_PROMPT if LEGACY_FORMAT else ENGLISH_CONNECT4_PROMPT
    return prompt.format(board=render(position["board"]), side=position["side"])

def extract_column(text: str) -> int | None:
    match = re.search(r"[1-7]", text)
    return int(match.group()) if match else None


def _number(text: str) -> str | None:
    match = re.search(r"-?\d[\d,]*", text)
    return match.group().replace(",", "") if match else None


def extract_gsm_answer(text: str) -> str | None:
    match = re.search(r"(?:Resposta|Answer)\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        number = _number(match.group(1))
        if number is not None:
            return number
    matches = re.findall(r"-?\d[\d,]*", text)
    return matches[-1].replace(",", "") if encontrados else None


def _load_model(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def _generate(model, tok, prompts: list[str], max_new_tokens: int, batch: int) -> list[str]:
    import torch

    responses = []
    for start in range(0, len(prompts), batch):
        batch_items = prompts[start:start + batch]
        messages = [[{"role": "user", "content": prompt}] for prompt in batch_items]
        texts = [tok.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True) for message in messages]
        inputs = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tok.eos_token_id)
        prompt_length = inputs.input_ids.shape[1]
        responses.extend(tok.decode(generated[i][prompt_length:], skip_special_tokens=True)
                         for i in range(len(batch_items)))
        print("  %d/%d" % (len(responses), len(prompts)), flush=True)
    return responses


def _evaluate_connect4(model, tok, args) -> dict:
    total = args.connect4
    by_type = {"win": (total + 1) // 2, "block": total // 2}
    result = {}
    all_results = []
    for kind, count in by_type.items():
        positions = generate_positions(count, kind, args.seed)
        texts = _generate(model, tok, [connect4_prompt(p) for p in positions], 8, args.batch)
        correct = sum(extract_column(t) in p["correct_columns"] for p, t in zip(positions, texts))
        chance = (sum(len(p["correct_columns"]) / len(legal_columns(p["board"]))
                     for p in positions) / len(positions)) if positions else 0.0
        result[kind] = {
            "n": len(positions), "correct": correct,
            "accuracy_pct": 100 * correct / len(positions) if positions else 0.0,
            "ci95": wilson(correct, len(positions)), "chance_pct": 100 * chance,
            "samples": [
                {"board": render(p["board"]), "side": p["side"],
                 "correct_columns": sorted(p["correct_columns"]), "answer": t}
                for p, t in zip(positions[:3], texts[:3])
            ],
        }
        all_results.extend(zip(positions, texts))
    total_correct = sum(extract_column(t) in p["correct_columns"] for p, t in all_results)
    total_n = len(all_results)
    total_chance = (sum(len(p["correct_columns"]) / len(legal_columns(p["board"]))
                       for p, _ in all_results) / total_n) if total_n else 0.0
    result["total"] = {
        "n": total_n, "correct": total_correct,
        "accuracy_pct": 100 * total_correct / total_n if total_n else 0.0,
        "ci95": wilson(total_correct, total_n), "chance_pct": 100 * total_chance,
    }
    return result


def _evaluate_gsm8k(model, tok, count: int, batch: int) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    items = list(dataset.select(range(min(max(0, count), len(dataset)))))
    prompts = [
        item["question"] + (LEGACY_GSM8K_SUFFIX if LEGACY_FORMAT else ENGLISH_GSM8K_SUFFIX)
        for item in items
    ]
    texts = _generate(model, tok, prompts, 320, batch)
    correct = 0
    samples = []
    for item, text in zip(items, texts):
        answer_key = item["answer"].split("####")[-1].strip()
        expected = _number(answer_key)
        predicted = extract_gsm_answer(text)
        correct += predicted is not None and predicted == expected
        if len(samples) < 3:
            samples.append({"question": item["question"], "expected": expected,
                             "extracted_answer": predicted, "text": text[:1000]})
    n = len(items)
    return {"n": n, "correct": correct,
            "accuracy_pct": 100 * correct / n if n else 0.0,
            "ci95": wilson(correct, n), "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--connect4", type=int, default=200)
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/generalization.json")
    args = ap.parse_args()
    if args.connect4 < 0 or args.gsm8k < 0 or args.batch < 1:
        ap.error("--connect4 and --gsm8k must be >= 0, and --batch must be >= 1")

    model, tok = _load_model(args.model, args.adapter)
    print("Evaluating Connect Four...", flush=True)
    connect4 = _evaluate_connect4(model, tok, args)
    print("Evaluating GSM8K...", flush=True)
    gsm8k = _evaluate_gsm8k(model, tok, args.gsm8k, args.batch)
    result = {"model": args.model, "adapter": args.adapter, "seed": args.seed,
                 "connect4": connect4, "gsm8k": gsm8k}

    print("\n=== GENERALIZATION SUMMARY ===")
    for kind in ("win", "block", "total"):
        r = connect4[kind]
        print("Connect Four %-8s: %d/%d = %.2f%% 95% CI [%.2f; %.2f], chance %.2f%%" % (
            kind, r["correct"], r["n"], r["accuracy_pct"], *r["ci95"],
            r["chance_pct"]))
    print("GSM8K      : %d/%d = %.2f%% 95% CI [%.2f; %.2f]" % (
        gsm8k["correct"], gsm8k["n"], gsm8k["accuracy_pct"], *gsm8k["ci95"]))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved to", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
