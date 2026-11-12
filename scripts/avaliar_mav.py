#!/usr/bin/env python3
"""Avalia o modelo de mundo (formato MAV) no split de TESTE.

Mede cada camada separadamente, na mesma resposta:
  - estado: a grade do tabuleiro após o lance anterior está exata?
            e por casa (inclui a COR, via maiúscula/minúscula)
  - regras: o conjunto de lances legais está exato? F1?
  - decisão: o melhor lance é legal? é a solução do puzzle?

O puzzle já vem no formato de transição: FEN anterior + lance do oponente.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr import mav
from chess_rlvr.dados import normalizar
from chess_rlvr.espelho import espelhar_tabuleiro, espelhar_uci
from chess_rlvr.verificador import montar_prompt, normalizar_uci


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, c - m), 100 * min(1.0, c + m))


def carregar(caminho: str, n: int, seed: int):
    linhas = []
    with open(caminho, encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            if i >= 50_000:
                break
            linhas.append(ln)
    random.Random(seed).shuffle(linhas)
    itens = []
    for ln in linhas:
        if len(itens) >= n:
            break
        try:
            l = normalizar(json.loads(ln))
            lances = l["moves"].split() if isinstance(l["moves"], str) else list(l["moves"])
            tab = chess.Board(l["fen"])
            tab.push_uci(lances[0])
        except Exception:
            continue
        itens.append({"fen0": l["fen"], "m0": lances[0], "tab": tab,
                      "solucao": lances[1], "rating": l["rating"]})
    return itens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="sem adapter = modelo base puro")
    ap.add_argument("--split", default="data/splits/teste.jsonl")
    ap.add_argument("--puzzles", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=440)
    ap.add_argument("--formato", choices=["mav", "direto"], default="mav",
                    help="direto = adapters antigos (so o lance), mesmos puzzles do teste")
    ap.add_argument("--salvar-texto", action="store_true",
                    help="guarda a resposta completa de cada item (analise de consistencia)")
    ap.add_argument("--verificar", action="store_true",
                    help="confere o lance final contra a posicao e cai para o top5/legais do proprio modelo")
    ap.add_argument("--espelho", action="store_true",
                    help="avalia os mesmos puzzles após espelhamento vertical")
    ap.add_argument("--saida", default="out/aval-mav.json")
    args = ap.parse_args()

    itens = carregar(args.split, args.puzzles, args.seed)
    descartados_espelho = 0
    if args.espelho:
        espelhados = []
        for x in itens:
            tab_inicial = espelhar_tabuleiro(chess.Board(x["fen0"]))
            m0 = espelhar_uci(x["m0"])
            solucao = espelhar_uci(x["solucao"])
            # `in legal_moves` com str levanta AttributeError; comparar Move
            if chess.Move.from_uci(m0) not in tab_inicial.legal_moves:
                descartados_espelho += 1
                continue
            tab = tab_inicial.copy()
            tab.push_uci(m0)
            if chess.Move.from_uci(solucao) not in tab.legal_moves:
                descartados_espelho += 1
                continue
            espelhados.append({**x, "fen0": tab_inicial.fen(), "m0": m0,
                               "solucao": solucao, "tab": tab})
        itens = espelhados
    if args.formato == "direto":
        args.max_tokens = 16
    print("puzzles do teste:", len(itens), flush=True)
    if args.espelho:
        print("descartados_espelho:", descartados_espelho, flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = AutoModelForCausalLM.from_pretrained(args.modelo, dtype=torch.bfloat16,
                                                  device_map="cuda:0")
    if args.adapter:
        modelo = PeftModel.from_pretrained(modelo, args.adapter)
    modelo.eval()

    textos = []
    for i in range(0, len(itens), args.lote):
        lote = itens[i:i + args.lote]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": (mav.prompt_transicao(x["fen0"], x["m0"])
                                          if args.formato == "mav" else montar_prompt(x["tab"]))}],
            tokenize=False, add_generation_prompt=True) for x in lote]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = modelo.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        for j in range(len(lote)):
            textos.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
        print("  %d/%d" % (len(textos), len(itens)), flush=True)

    c = {"grade_exata": 0, "casas_certas": 0, "casas_total": 0,
         "cor_certa": 0, "cor_total": 0, "legais_exatos": 0, "f1_soma": 0.0,
         "melhor_legal": 0, "melhor_correto": 0}
    amostras = []
    registros = []
    for x, texto in zip(itens, textos):
        tab = x["tab"]
        campos = (mav.extrair_campos(texto) if args.formato == "mav"
                  else {"grade": None, "legais": None, "melhor": normalizar_uci(texto)})
        certa = mav.grade(tab)
        if campos["grade"] == certa:
            c["grade_exata"] += 1
        if campos["grade"]:
            for lg, lc in zip(campos["grade"].splitlines()[:8], certa.splitlines()[:8]):
                for a, b in zip(lg.split()[1:], lc.split()[1:]):
                    c["casas_total"] += 1
                    c["casas_certas"] += a == b
                    if b != ".":
                        c["cor_total"] += 1
                        c["cor_certa"] += a != "." and a.isupper() == b.isupper()
        verdade = {m.uci() for m in tab.legal_moves}
        prev = campos["legais"] or set()
        if prev == verdade:
            c["legais_exatos"] += 1
        tp = len(prev & verdade)
        if prev and tp:
            prec, rec = tp / len(prev), tp / len(verdade)
            c["f1_soma"] += 2 * prec * rec / (prec + rec)
        m = campos["melhor"]
        if args.verificar:
            m, _origem = mav.lance_verificado(texto, tab)
        if m and m in verdade:
            c["melhor_legal"] += 1
        correto = m == x["solucao"]
        legal = bool(m and m in verdade)
        if correto:
            c["melhor_correto"] += 1
        registro = {"rating": int(x["rating"]), "solucao": x["solucao"],
                    "melhor": m, "correto": correto, "legal": legal,
                    "grade_exata": (campos["grade"] == certa
                                    if args.formato == "mav" else None)}
        if args.salvar_texto:
            registro["texto"] = texto
        registros.append(registro)
        if len(amostras) < 3:
            amostras.append({"esperado": x["solucao"], "texto": texto[:700]})

    n = len(itens)
    res = {
        "n": n, "adapter": args.adapter, "verificado": args.verificar,
        "itens": registros,
        "grade_exata_pct": 100 * c["grade_exata"] / n,
        "casas_certas_pct": 100 * c["casas_certas"] / max(1, c["casas_total"]),
        "cor_certa_pct": 100 * c["cor_certa"] / max(1, c["cor_total"]),
        "legais_exatos_pct": 100 * c["legais_exatos"] / n,
        "legais_f1_medio": c["f1_soma"] / n,
        "melhor_legal_pct": 100 * c["melhor_legal"] / n,
        "melhor_correto": c["melhor_correto"],
        "melhor_correto_pct": 100 * c["melhor_correto"] / n,
        "melhor_correto_ic95": wilson(c["melhor_correto"], n),
        "amostras": amostras,
        "registros": registros,
    }
    if args.espelho:
        res["descartados_espelho"] = descartados_espelho
    print()
    print("=== MODELO DE MUNDO (teste, n=%d) ===" % n)
    print("estado  grade exata      : %6.2f%%" % res["grade_exata_pct"])
    print("estado  casas certas     : %6.2f%%" % res["casas_certas_pct"])
    print("estado  cor certa        : %6.2f%%" % res["cor_certa_pct"])
    print("regras  legais exatos    : %6.2f%%" % res["legais_exatos_pct"])
    print("regras  legais F1 medio  : %6.3f" % res["legais_f1_medio"])
    print("decisao melhor e legal   : %6.2f%%" % res["melhor_legal_pct"])
    print("decisao melhor = solucao : %d/%d = %.2f%% IC95 [%.2f; %.2f]" % (
        c["melhor_correto"], n, res["melhor_correto_pct"], *res["melhor_correto_ic95"]))
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
