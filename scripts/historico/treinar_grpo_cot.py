#!/usr/bin/env python3
"""GRPO sobre o RACIOCÍNIO (RLVR), estilo R1-Zero.

Diferença crucial em relação ao GRPO de resposta direta:
  - a ação do modelo é o traço inteiro (candidatos + comparação + lance)
  - a recompensa continua vindo do lance final, que é verificável
  - o modelo descobre a própria estratégia de raciocínio; não copia template

Isto é o que a literatura chama de RLVR: recompensa verificável, política livre.
O formato é premiado separadamente para não colapsar em saída curta.

Custo de inferência também é medido: raciocínio longo custa tokens.
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
from chess_rlvr.raciocinio import prompt_candidatos, prompt_forcados
from chess_rlvr.stockfish import Motor, localizar_motor, normalizar_vantagem
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle

_MOTOR = None
_RE_LANCE = re.compile(r"LANCE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
_RE_UCI = re.compile(r"\b([a-h][1-8])[- ]?([a-h][1-8])([qrbn])?\b", re.IGNORECASE)
_RE_CAND = re.compile(r"^\s*[-*]\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.MULTILINE)


def _motor():
    global _MOTOR
    if _MOTOR is None:
        caminho = localizar_motor()
        if caminho is None:
            raise RuntimeError("Stockfish ausente")
        _MOTOR = Motor(caminho, profundidade=6).__enter__()
    return _MOTOR


def extrair_lance(texto: str) -> str | None:
    """Prefere a linha `LANCE:`; cai para o primeiro UCI válido."""
    m = _RE_LANCE.search(texto or "")
    if m:
        return m.group(1).lower()
    m = _RE_UCI.search(texto or "")
    if m:
        return (m.group(1) + m.group(2) + (m.group(3) or "")).lower()
    return None


def construir(quantidade: int, seed: int, rating_max: int | None,
              formato: str = "candidatos") -> Dataset:
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    prompts, fens, solucoes = [], [], []
    for bruto in ds:
        if len(prompts) >= quantidade:
            break
        l = normalizar(bruto)
        try:
            p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                              rating=l["rating"], temas=l["temas"], game_id=l["game_id"])
        except PuzzleInvalido:
            continue
        if rating_max and p.rating > rating_max:
            continue
        prompts.append(prompt_forcados(p.tabuleiro) if formato == "forcados"
                       else prompt_candidatos(p.tabuleiro))
        fens.append(p.tabuleiro.fen())
        solucoes.append(p.primeiro_lance)
    return Dataset.from_dict({"prompt": prompts, "fen": fens, "solucao": solucoes})


# ---------------------------------------------------------------------------
# Recompensas
# ---------------------------------------------------------------------------

def recompensa_acerto(prompts, completions, fen=None, solucao=None, **kwargs):
    """O sinal principal: o lance extraído bate com a solução."""
    out = []
    for texto, alvo in zip(completions, solucao or []):
        lance = extrair_lance(texto)
        if lance is None:
            out.append(-0.5)
        elif lance == alvo:
            out.append(1.0)
        else:
            out.append(0.0)
    return out


def recompensa_formato(prompts, completions, **kwargs):
    """Premia a estrutura de raciocínio sem exigir um template exato.

    Sem isso, o RL colapsa em respostas curtas: emitir só o lance é mais barato
    e às vezes acerta igual. Com isso, manter o raciocínio tem valor próprio.
    """
    out = []
    for texto in completions:
        t = (texto or "").lower()
        pontos = 0.0
        if "situa" in t:
            pontos += 0.05
        if "candidat" in t:
            pontos += 0.05
        if "compara" in t:
            pontos += 0.05
        if _RE_LANCE.search(texto or ""):
            pontos += 0.05
        out.append(pontos)
    return out


def recompensa_candidatos(prompts, completions, solucao=None, **kwargs):
    """+0.5 se a solução apareceu entre os candidatos listados.

    Esta é a recompensa que ataca o gargalo medido. Separando as duas
    habilidades: quando a solução estava na lista o modelo acertou 74,03%;
    quando não estava, acertou 0 de 523. O problema é a BUSCA (12,83% de
    recall), não a escolha.

    Premiar só o lance final não diz nada sobre "colocar o certo na lista":
    o modelo pode listar três lances ruins, escolher o menos ruim e receber a
    mesma recompensa de quem listou o certo.
    """
    saida = []
    for texto, alvo in zip(completions, solucao or []):
        candidatos = [c.lower() for c in _RE_CAND.findall(texto or "")]
        saida.append(0.5 if alvo in candidatos else 0.0)
    return saida


def recompensa_legalidade(prompts, completions, fen=None, **kwargs):
    out = []
    for texto, f in zip(completions, fen or []):
        lance = extrair_lance(texto)
        if lance is None:
            out.append(-0.3)
            continue
        try:
            tab = chess.Board(f)
            mv = chess.Move.from_uci(lance)
            out.append(0.1 if mv in tab.legal_moves else -0.3)
        except Exception:
            out.append(-0.3)
    return out


def recompensa_stockfish(prompts, completions, fen=None, solucao=None, **kwargs):
    motor = _motor()
    out = []
    for texto, f, alvo in zip(completions, fen or [], solucao or []):
        lance = extrair_lance(texto)
        if lance is None:
            out.append(-1.0)
            continue
        if lance == alvo:
            out.append(1.0)
            continue
        try:
            tab = chess.Board(f)
            mv = chess.Move.from_uci(lance)
            if mv not in tab.legal_moves:
                out.append(-1.0)
                continue
            out.append(normalizar_vantagem(motor.avaliar_apos(tab, mv)))
        except Exception:
            out.append(-1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="out/sft-cot2")
    ap.add_argument("--amostras", type=int, default=2000)
    ap.add_argument("--rating-max", type=int, default=1400)
    ap.add_argument("--passos", type=int, default=150)
    ap.add_argument("--geracoes", type=int, default=8)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--acum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--max-completude", type=int, default=192)
    ap.add_argument("--4bit", dest="quatro_bit", action="store_true")
    ap.add_argument("--usar-stockfish", action="store_true")
    ap.add_argument("--recompensa-candidatos", action="store_true",
                    help="premia listar a solucao entre os candidatos (ataca a busca)")
    ap.add_argument("--formato", choices=["candidatos", "forcados"], default="candidatos")
    ap.add_argument("--temperatura", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/grpo-cot")
    args = ap.parse_args()

    ds = construir(args.amostras, args.seed, args.rating_max, args.formato)
    print("amostras:", len(ds), "| rating_max:", args.rating_max,
          "| formato:", args.formato, flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.quatro_bit:
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo, quantization_config=qc, device_map="cuda:0")
    else:
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo, dtype=torch.bfloat16, device_map="cuda:0")

    partir = bool(args.adapter and Path(args.adapter).exists())
    if partir:
        from peft import PeftModel
        print("partindo do adapter:", args.adapter, flush=True)
        modelo = PeftModel.from_pretrained(modelo, args.adapter, is_trainable=True)

    recompensas = [recompensa_acerto, recompensa_formato, recompensa_legalidade]
    if args.recompensa_candidatos:
        recompensas.append(recompensa_candidatos)
        print("recompensa de CANDIDATOS ATIVA (ataca a busca)", flush=True)
    if args.usar_stockfish:
        recompensas.append(recompensa_stockfish)
        print("recompensa Stockfish ATIVA", flush=True)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    cfg = GRPOConfig(
        output_dir=args.saida, max_steps=args.passos,
        per_device_train_batch_size=args.lote,
        gradient_accumulation_steps=args.acum, learning_rate=args.lr,
        num_generations=args.geracoes,
        max_completion_length=args.max_completude,
        logging_steps=10, save_strategy="no", bf16=True,
        gradient_checkpointing=True, report_to=[], seed=args.seed,
        temperature=args.temperatura, beta=0.04, loss_type="grpo",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=modelo, reward_funcs=recompensas, args=cfg, train_dataset=ds,
        peft_config=None if partir else lora, processing_class=tok)

    print("treinando GRPO sobre raciocinio...", flush=True)
    trainer.train()
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    trainer.save_model(args.saida)

    Path("out").mkdir(exist_ok=True)
    Path("out/grpo-cot-metricas.json").write_text(json.dumps({
        "amostras": len(ds), "passos": args.passos, "geracoes": args.geracoes,
        "max_completude": args.max_completude, "quatro_bit": args.quatro_bit,
        "usar_stockfish": args.usar_stockfish, "rating_max": args.rating_max,
        "recompensa_candidatos": args.recompensa_candidatos,
        "formato": args.formato,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": trainer.state.log_history[-15:],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
