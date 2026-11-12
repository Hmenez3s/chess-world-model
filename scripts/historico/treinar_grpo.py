#!/usr/bin/env python3
"""GRPO sobre o adapter de SFT.

Ponto de partida: SFT com 48,5% de lance legal e 4,5% de acerto. Isso dá sinal para
o GRPO (sem SFT, 0% de acerto = gradiente relativo nulo).

Desenho da recompensa, deliberadamente em duas parcelas:
  - legalidade: sinaliza o que o formato já ensinou, mas evita que o modelo regrida
  - correção:   é o que o RL precisa descobrir sozinho

A recompensa olha SÓ o lance, nunca o texto de raciocínio. Se recompensássemos o
raciocínio, o modelo aprenderia a escrever texto plausível — e a correlação entre
raciocínio e acerto deixaria de ser uma medida.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from chess_rlvr.dados import normalizar
from chess_rlvr.stockfish import Motor, localizar_motor, normalizar_vantagem
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle, montar_prompt

# Motor unico e reutilizado: abrir Stockfish por lance seria proibitivo.
_MOTOR = None


def _motor():
    global _MOTOR
    if _MOTOR is None:
        caminho = localizar_motor()
        if caminho is None:
            raise RuntimeError("Stockfish nao encontrado; recompensa parcial indisponivel")
        _MOTOR = Motor(caminho, profundidade=6).__enter__()
    return _MOTOR

_PADRAO_UCI = re.compile(r"\b([a-h][1-8])[- ]?([a-h][1-8])([qrbn])?\b", re.IGNORECASE)


def extrair_uci(texto: str) -> str | None:
    m = _PADRAO_UCI.search(texto or "")
    if not m:
        return None
    return (m.group(1) + m.group(2) + (m.group(3) or "")).lower()


def construir_dataset(quantidade: int, seed: int, rating_max: int | None = None) -> Dataset:
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    prompts, fens, solucoes, ratings = [], [], [], []
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
        if rating_max is not None and p.rating > rating_max:
            continue
        prompts.append(montar_prompt(p.tabuleiro))
        fens.append(p.tabuleiro.fen())
        solucoes.append(p.primeiro_lance)
        ratings.append(p.rating)

    return Dataset.from_dict({"prompt": prompts, "fen": fens,
                              "solucao": solucoes, "rating": ratings})


# ---------------------------------------------------------------------------
# Funções de recompensa — assinatura exigida pelo GRPOTrainer
# ---------------------------------------------------------------------------

def recompensa_legalidade(prompts, completions, fen=None, **kwargs):
    """+0.2 se o lance é legal, -0.5 se é ilegal ou ilegível.

    Mantém o piso de legalidade que o SFT conquistou; sem isso o RL pode
    trocar acerto por sintaxe inválida, um modo de falha comum.
    """
    saida = []
    for texto, f in zip(completions, fen or []):
        uci = extrair_uci(texto)
        if uci is None:
            saida.append(-0.5)
            continue
        try:
            tab = chess.Board(f)
            mv = chess.Move.from_uci(uci)
            saida.append(0.2 if mv in tab.legal_moves else -0.5)
        except Exception:
            saida.append(-0.5)
    return saida


def recompensa_stockfish(prompts, completions, fen=None, solucao=None, **kwargs):
    """Recompensa continua: o quao boa e a posicao apos o lance.

    Esta e a parcela que da sinal onde a binaria nao da. Um lance legal que nao e a
    solucao mas mantem a vantagem recebe recompensa positiva; um lance que entrega
    a posicao recebe negativa. Sem isso, ~90% dos grupos tem variancia zero.
    """
    motor = _motor()
    saida = []
    for texto, f, alvo in zip(completions, fen or [], solucao or []):
        uci = extrair_uci(texto)
        if uci is None:
            saida.append(-1.0)
            continue
        if uci == alvo:
            saida.append(1.0)
            continue
        try:
            tab = chess.Board(f)
            mv = chess.Move.from_uci(uci)
            if mv not in tab.legal_moves:
                saida.append(-1.0)
                continue
            cp = motor.avaliar_apos(tab, mv)
            saida.append(normalizar_vantagem(cp))
        except Exception:
            saida.append(-1.0)
    return saida


def recompensa_correcao(prompts, completions, fen=None, solucao=None, **kwargs):
    """+1.0 acerto exato; +0.1 lance legal mas errado; -0.2 ilegal.

    O +0.1 no lance legal-errado é o que evita o colapso: com recompensa
    puramente binária e ~4,5% de acerto, quase todo grupo fica com variância zero
    e o modelo não aprende.
    """
    saida = []
    for texto, f, alvo in zip(completions, fen or [], solucao or []):
        uci = extrair_uci(texto)
        if uci is None:
            saida.append(-0.2)
            continue
        if uci == alvo:
            saida.append(1.0)
            continue
        try:
            tab = chess.Board(f)
            mv = chess.Move.from_uci(uci)
            saida.append(0.1 if mv in tab.legal_moves else -0.2)
        except Exception:
            saida.append(-0.2)
    return saida


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="out/sft-adapter")
    ap.add_argument("--amostras", type=int, default=3000)
    ap.add_argument("--4bit", dest="quatro_bit", action="store_true",
                    help="QLoRA: carrega o modelo em 4-bit (config pedida no plano)")
    ap.add_argument("--usar-stockfish", action="store_true",
                    help="Adiciona recompensa parcial via Stockfish")
    ap.add_argument("--rating-max", type=int, default=None,
                    help="Curriculo: treina so com puzzles ate este rating")
    ap.add_argument("--passos", type=int, default=300)
    ap.add_argument("--geracoes", type=int, default=8)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--acum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-completude", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/grpo-adapter")
    args = ap.parse_args()

    print("construindo dataset...", flush=True)
    ds = construir_dataset(args.amostras, args.seed, args.rating_max)
    print("amostras:", len(ds), "rating_max:", args.rating_max, flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.quatro_bit:
        from transformers import BitsAndBytesConfig
        print("carregando em 4-bit (QLoRA)", flush=True)
        qc = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo, quantization_config=qc, device_map="cuda:0")
    else:
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo, dtype=torch.bfloat16, device_map="cuda:0")

    # Se partimos de um adapter, continuamos NELE: o GRPOTrainer recusa
    # PeftModel + peft_config. Assim o SFT e o ponto de partida real do RL.
    partir_de_adapter = bool(args.adapter and Path(args.adapter).exists())
    if partir_de_adapter:
        from peft import PeftModel
        print("partindo do adapter SFT:", args.adapter, flush=True)
        modelo = PeftModel.from_pretrained(modelo, args.adapter, is_trainable=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # A recompensa de Stockfish e cara; entra so quando pedida.
    recompensas = [recompensa_correcao, recompensa_legalidade]
    if args.usar_stockfish:
        recompensas.append(recompensa_stockfish)
        print("recompensa Stockfish ATIVA (depth 6)", flush=True)

    cfg = GRPOConfig(
        output_dir=args.saida,
        max_steps=args.passos,
        per_device_train_batch_size=args.lote,
        gradient_accumulation_steps=args.acum,
        learning_rate=args.lr,
        num_generations=args.geracoes,
        max_completion_length=args.max_completude,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        temperature=1.0,
        beta=0.04,
        loss_type="grpo",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=modelo, reward_funcs=recompensas,
        args=cfg, train_dataset=ds,
        peft_config=None if partir_de_adapter else lora,
        processing_class=tok,
    )

    print("treinando GRPO...", flush=True)
    trainer.train()
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

    trainer.save_model(args.saida)
    Path("out").mkdir(exist_ok=True)
    Path("out/grpo-metricas.json").write_text(json.dumps({
        "amostras": len(ds), "passos": args.passos, "geracoes": args.geracoes,
        "rating_max": args.rating_max, "usar_stockfish": args.usar_stockfish,
        "quatro_bit": args.quatro_bit,
        "lote": args.lote, "acum": args.acum, "lr": args.lr,
        "max_completude": args.max_completude,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": trainer.state.log_history[-15:],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("adapter GRPO salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
