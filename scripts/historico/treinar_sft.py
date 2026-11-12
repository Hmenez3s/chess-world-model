#!/usr/bin/env python3
"""SFT de warmup: ensina o formato e o vocabulário de lances.

Por que existe: com 5% de lance legal e 0% de acerto, todo rollout do GRPO é
negativo, o gradiente relativo é nulo e o modelo não sai do lugar. O SFT dá o
ponto de partida mínimo para que o RL tenha sinal.

Deliberadamente NÃO ensinamos raciocínio aqui: o alvo é só o lance. Assim, qualquer
ganho do GRPO é atribuível ao RL e não ao formato do SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from chess_rlvr.dados import normalizar
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle, montar_prompt


def construir_dataset(quantidade: int, seed: int):
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    prompts, respostas = [], []
    for bruto in ds:
        if len(prompts) >= quantidade:
            break
        linha = normalizar(bruto)
        try:
            p = montar_puzzle(
                puzzle_id=linha["puzzle_id"], fen=linha["fen"], moves=linha["moves"],
                rating=linha["rating"], temas=linha["temas"], game_id=linha["game_id"],
            )
        except PuzzleInvalido:
            continue
        prompts.append(montar_prompt(p.tabuleiro))
        respostas.append(p.primeiro_lance)

    from datasets import Dataset
    # Formato conversacional: o template e aplicado uma vez sobre a conversa
    # inteira, evitando a divergencia "prompt nao e prefixo de prompt+completion"
    # que treinaria o modelo num contexto diferente do de inferencia.
    mensagens = [
        [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        for p, r in zip(prompts, respostas)
    ]
    return Dataset.from_dict({"messages": mensagens})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--amostras", type=int, default=15000)
    ap.add_argument("--epocas", type=float, default=1.0)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--acum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/sft-adapter")
    args = ap.parse_args()

    print("construindo dataset...", flush=True)
    ds = construir_dataset(args.amostras, args.seed)
    print("amostras:", len(ds), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map="cuda:0",
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.saida,
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.lote,
        gradient_accumulation_steps=args.acum,
        learning_rate=args.lr,
        logging_steps=25,
        save_strategy="no",
        bf16=True,
        max_length=1024,
        completion_only_loss=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=modelo, args=cfg, train_dataset=ds,
        peft_config=lora, processing_class=tok,
    )

    print("treinando...", flush=True)
    trainer.train()

    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    trainer.save_model(args.saida)
    tok.save_pretrained(args.saida)

    Path("out").mkdir(exist_ok=True)
    Path("out/sft-metricas.json").write_text(json.dumps({
        "amostras": len(ds), "epocas": args.epocas, "lote": args.lote,
        "acum": args.acum, "lr": args.lr,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "estado_log": trainer.state.log_history[-5:],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("adapter salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
