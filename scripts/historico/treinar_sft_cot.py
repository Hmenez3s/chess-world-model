#!/usr/bin/env python3
"""SFT de raciocínio: treina o modelo a produzir traços verificáveis.

Diferença crucial em relação ao SFT anterior: aqui o alvo é um raciocínio em três
partes, construído a partir de fatos do tabuleiro — não a resposta seca.

Sem este passo, qualquer medição de "raciocínio ajuda?" mede apenas alucinação:
o modelo nunca aprendeu o formato.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from chess_rlvr.dados import normalizar
from chess_rlvr.raciocinio import gerar_traco, traco_para_prompt
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle


def construir(quantidade: int, seed: int) -> Dataset:
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    mensagens = []
    for bruto in ds:
        if len(mensagens) >= quantidade:
            break
        l = normalizar(bruto)
        try:
            p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                              rating=l["rating"], temas=l["temas"], game_id=l["game_id"])
        except PuzzleInvalido:
            continue
        lance = p.primeiro_lance
        import chess
        traco = gerar_traco(p.tabuleiro, chess.Move.from_uci(lance))
        mensagens.append([
            {"role": "user", "content": traco_para_prompt(p.tabuleiro)},
            {"role": "assistant", "content": traco},
        ])

    return Dataset.from_dict({"messages": mensagens})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--amostras", type=int, default=8000)
    ap.add_argument("--epocas", type=float, default=1.0)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--acum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/sft-cot")
    args = ap.parse_args()

    print("construindo dataset de raciocinio...", flush=True)
    ds = construir(args.amostras, args.seed)
    print("amostras:", len(ds), flush=True)
    print("exemplo de alvo:", flush=True)
    print(ds[0]["messages"][1]["content"][:400], flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map="cuda:0")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(
        output_dir=args.saida, num_train_epochs=args.epocas,
        per_device_train_batch_size=args.lote,
        gradient_accumulation_steps=args.acum, learning_rate=args.lr,
        logging_steps=25, save_strategy="no", bf16=True, max_length=1024,
        completion_only_loss=True, gradient_checkpointing=True,
        report_to=[], seed=args.seed, dataloader_num_workers=2,
    )

    trainer = SFTTrainer(model=modelo, args=cfg, train_dataset=ds,
                          peft_config=lora, processing_class=tok)
    print("treinando SFT de raciocinio...", flush=True)
    trainer.train()
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    trainer.save_model(args.saida)
    tok.save_pretrained(args.saida)

    Path("out").mkdir(exist_ok=True)
    Path("out/sft-cot-metricas.json").write_text(json.dumps({
        "amostras": len(ds), "epocas": args.epocas,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": trainer.state.log_history[-4:],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
