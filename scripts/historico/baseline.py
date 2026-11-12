#!/usr/bin/env python3
"""Mede o modelo BASE antes de qualquer treino.

Sem este número não existe experimento: não há como afirmar que GRPO melhorou nada.
Roda também a versão espelhada, para estabelecer desde já a linha de base do teste
de memorização (um modelo não treinado não pode ter memorizado, então espelho ≈ original
aqui é a expectativa — se divergir muito, o espelho tem defeito, não o modelo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr.avaliacao import ResultadoAvaliacao
from chess_rlvr.dados import faixa_de_rating, normalizar
from chess_rlvr.espelho import EspelhoInvalido, espelhar_puzzle
from chess_rlvr.verificador import (
    PuzzleInvalido,
    avaliar_lance,
    montar_puzzle,
    montar_prompt,
    normalizar_uci,
)


def carregar_puzzles(quantidade: int, seed: int):
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    saida = []
    for bruto in ds:
        if len(saida) >= quantidade:
            break
        linha = normalizar(bruto)
        try:
            p = montar_puzzle(
                puzzle_id=linha["puzzle_id"], fen=linha["fen"], moves=linha["moves"],
                rating=linha["rating"], temas=linha["temas"], game_id=linha["game_id"],
            )
        except PuzzleInvalido:
            continue
        saida.append(p)
    return saida


def gerar(modelo, tok, prompts: list[str], device: str, max_novos: int, lote: int):
    respostas = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), lote):
        bloco = prompts[i:i + lote]
        msgs = [[{"role": "user", "content": p}] for p in bloco]
        entradas = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to(device)
        with torch.no_grad():
            saida = modelo.generate(
                **entradas, max_new_tokens=max_novos, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        novos = saida[:, entradas["input_ids"].shape[1]:]
        respostas.extend(tok.batch_decode(novos, skip_special_tokens=True))
        print("  gerados %d/%d" % (min(i + lote, len(prompts)), len(prompts)), flush=True)
    return respostas


def avaliar(modelo, tok, puzzles, device, max_novos, lote, rotulo):
    prompts = [montar_prompt(p.tabuleiro) for p in puzzles]
    inicio = time.time()
    respostas = gerar(modelo, tok, prompts, device, max_novos, lote)
    duracao = time.time() - inicio

    res = ResultadoAvaliacao()
    for p, texto in zip(puzzles, respostas):
        lance = normalizar_uci(texto)
        r = avaliar_lance(tabuleiro=p.tabuleiro, lance_previsto=lance,
                          lance_solucao=p.primeiro_lance)
        res.registrar(status=r.status, faixa=faixa_de_rating(p.rating), temas=p.temas)

    print()
    print(res.relatorio(rotulo))
    print("tempo: %.1fs (%.2f s/puzzle)" % (duracao, duracao / max(1, len(puzzles))))
    return res, respostas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-novos", type=int, default=16)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--saida", default="out/baseline.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print("gpu0:", p.name, "%.1fGB" % (p.total_memory / 1e9))

    print("carregando puzzles...", flush=True)
    puzzles = carregar_puzzles(args.puzzles, args.seed)
    print("puzzles:", len(puzzles))

    print("carregando modelo...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map=device,
    )
    modelo.eval()
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

    res, respostas = avaliar(modelo, tok, puzzles, device, args.max_novos, args.lote,
                             "BASELINE — posicao original")

    # Espelho: mesma tática, coordenadas trocadas.
    espelhados, originais = [], []
    for p in puzzles:
        try:
            espelhados.append(espelhar_puzzle(p))
            originais.append(p)
        except EspelhoInvalido:
            continue
    print("\nespelhados disponiveis:", len(espelhados))
    res_e, _ = avaliar(modelo, tok, espelhados, device, args.max_novos, args.lote,
                       "BASELINE — posicao espelhada")

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps({
        "modelo": args.modelo,
        "puzzles": len(puzzles),
        "seed": args.seed,
        "max_novos": args.max_novos,
        "original": res.como_dict(),
        "espelhado": res_e.como_dict(),
        "amostra_respostas": respostas[:10],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsalvo em", saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
