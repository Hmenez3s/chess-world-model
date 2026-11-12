#!/usr/bin/env python3
"""Avalia um modelo (base ou com adapter) no mesmo contrato do baseline.

Roda posição original e espelhada. A diferença entre elas é a medida de memorização:
um modelo que aprendeu a ler o tabuleiro mantém a taxa; um que decorou despenca.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr.avaliacao import ResultadoAvaliacao
from chess_rlvr.dados import faixa_de_rating, normalizar
from chess_rlvr.espelho import EspelhoInvalido, espelhar_puzzle
from chess_rlvr.verificador import (
    PuzzleInvalido, avaliar_lance, montar_puzzle, montar_prompt, normalizar_uci,
)


def carregar_puzzles(quantidade: int, seed: int, deslocamento: int = 0):
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    saida = []
    vistos = 0
    for bruto in ds:
        linha = normalizar(bruto)
        try:
            p = montar_puzzle(
                puzzle_id=linha["puzzle_id"], fen=linha["fen"], moves=linha["moves"],
                rating=linha["rating"], temas=linha["temas"], game_id=linha["game_id"],
            )
        except PuzzleInvalido:
            continue
        vistos += 1
        if vistos <= deslocamento:
            continue
        saida.append(p)
        if len(saida) >= quantidade:
            break
    return saida


def gerar(modelo, tok, prompts, device, max_novos, lote):
    respostas = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), lote):
        bloco = prompts[i:i + lote]
        msgs = [[{"role": "user", "content": p}] for p in bloco]
        ent = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", padding=True,
                                      return_dict=True).to(device)
        with torch.no_grad():
            out = modelo.generate(**ent, max_new_tokens=max_novos, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        respostas.extend(tok.batch_decode(out[:, ent["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
    return respostas


def avaliar(modelo, tok, puzzles, device, max_novos, lote, rotulo):
    inicio = time.time()
    respostas = gerar(modelo, tok, [montar_prompt(p.tabuleiro) for p in puzzles],
                      device, max_novos, lote)
    dur = time.time() - inicio
    res = ResultadoAvaliacao()
    for p, texto in zip(puzzles, respostas):
        r = avaliar_lance(tabuleiro=p.tabuleiro,
                          lance_previsto=normalizar_uci(texto),
                          lance_solucao=p.primeiro_lance)
        res.registrar(status=r.status, faixa=faixa_de_rating(p.rating), temas=p.temas)
    print()
    print(res.relatorio(rotulo))
    print("tempo: %.1fs (%.3f s/puzzle)" % (dur, dur / max(1, len(puzzles))))
    return res, respostas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-novos", type=int, default=16)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--rotulo", default="AVALIACAO")
    ap.add_argument("--saida", default="out/avaliacao.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print("gpu0:", p.name, "%.1fGB" % (p.total_memory / 1e9))

    print("carregando puzzles...", flush=True)
    puzzles = carregar_puzzles(args.puzzles, args.seed)
    print("puzzles:", len(puzzles), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map=device)
    if args.adapter:
        from peft import PeftModel
        print("aplicando adapter:", args.adapter, flush=True)
        modelo = PeftModel.from_pretrained(modelo, args.adapter)
    modelo.eval()

    res, respostas = avaliar(modelo, tok, puzzles, device, args.max_novos, args.lote,
                             args.rotulo + " — original")

    esp, orig = [], []
    for p in puzzles:
        try:
            esp.append(espelhar_puzzle(p)); orig.append(p)
        except EspelhoInvalido:
            continue
    res_e, _ = avaliar(modelo, tok, esp, device, args.max_novos, args.lote,
                       args.rotulo + " — espelhado")

    # Queda relativa: quanto o desempenho depende das coordenadas exatas.
    queda = (res.correto.taxa - res_e.correto.taxa) / res.correto.taxa \
        if res.correto.taxa > 0 else 0.0

    print()
    print("=== RESUMO %s ===" % args.rotulo)
    print("original  : %d/%d correto (%.2f%%)" % (
        res.correto.sucessos, res.correto.total, 100 * res.correto.taxa))
    print("espelhado : %d/%d correto (%.2f%%)" % (
        res_e.correto.sucessos, res_e.correto.total, 100 * res_e.correto.taxa))
    print("queda relativa no espelho: %.1f%%" % (100 * queda))
    print("VRAM pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps({
        "modelo": args.modelo, "adapter": args.adapter,
        "puzzles": len(puzzles), "seed": args.seed,
        "original": res.como_dict(), "espelhado": res_e.como_dict(),
        "queda_relativa_espelho": round(queda, 4),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "amostra_respostas": respostas[:10],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
