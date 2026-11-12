#!/usr/bin/env python3
"""SFT de raciocínio sobre o adapter que já sabe responder.

Correção metodológica em relação à primeira tentativa:

  ANTES: treinava raciocínio a partir do modelo BASE. O modelo nunca tinha
         aprendido a responder, então comparar com o SFT direto era inválido —
         um braço tinha a habilidade de resposta, o outro não.

  AGORA: parte do adapter de movimentos e ADICIONA o raciocínio por cima.
         Assim o único fator que muda é o formato de raciocínio, e a comparação
         passa a ser causal.

Dados: traços com exploração de candidatos, verificados por rejection sampling.
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


def carregar_tracos(caminho: str, limite: int | None) -> Dataset:
    mensagens = []
    with open(caminho, encoding="utf-8") as fh:
        for linha in fh:
            if limite and len(mensagens) >= limite:
                break
            d = json.loads(linha)
            mensagens.append([
                {"role": "user", "content": d["prompt"]},
                {"role": "assistant", "content": d["resposta"]},
            ])
    return Dataset.from_dict({"messages": mensagens})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-base", default="out/sft-adapter",
                    help="Adapter de movimentos; o raciocinio e adicionado sobre ele")
    ap.add_argument("--tracos", default="data/tracos.jsonl")
    ap.add_argument("--amostras", type=int, default=0, help="0 = todas")
    ap.add_argument("--epocas", type=float, default=1.0)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--acum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=64,
                    help="rank do LoRA novo quando nao ha adapter base")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-passos", type=int, default=-1, help="-1 = epocas inteiras")
    ap.add_argument("--checkpointing", type=int, default=1, help="1 liga, 0 desliga")
    ap.add_argument("--saida", default="out/sft-cot2")
    args = ap.parse_args()

    ds = carregar_tracos(args.tracos, args.amostras or None)
    print("tracos carregados:", len(ds), flush=True)
    print("exemplo:", ds[0]["messages"][1]["content"][:260].replace("\n", " | "), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map="cuda:0")

    if args.adapter_base and Path(args.adapter_base).exists():
        print("partindo do adapter:", args.adapter_base, flush=True)
        modelo = PeftModel.from_pretrained(modelo, args.adapter_base, is_trainable=True)
    else:
        print("LoRA novo r=%d sobre o modelo base" % args.lora_r, flush=True)
        modelo = get_peft_model(modelo, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    cfg = SFTConfig(
        output_dir=args.saida, num_train_epochs=args.epocas,
        per_device_train_batch_size=args.lote,
        gradient_accumulation_steps=args.acum, learning_rate=args.lr,
        logging_steps=25, save_strategy="no", bf16=True, max_length=args.max_length,
        max_steps=args.max_passos,
        completion_only_loss=True, gradient_checkpointing=bool(args.checkpointing),
        report_to=[], seed=args.seed, dataloader_num_workers=2,
    )

    trainer = SFTTrainer(model=modelo, args=cfg, train_dataset=ds,
                          processing_class=tok)
    print("treinando raciocinio sobre o adapter...", flush=True)
    trainer.train()
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    trainer.save_model(args.saida)
    tok.save_pretrained(args.saida)

    Path("out").mkdir(exist_ok=True)
    # nome derivado da saida, para nao sobrescrever a metrica de outra execucao
    nome_metrica = Path("out") / (Path(args.saida).name + "-metricas.json")
    nome_metrica.write_text(json.dumps({
        "tracos": len(ds), "epocas": args.epocas, "lr": args.lr,
        "adapter_base": args.adapter_base,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": trainer.state.log_history[-4:],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("salvo em", args.saida, "| metricas em", nome_metrica)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
