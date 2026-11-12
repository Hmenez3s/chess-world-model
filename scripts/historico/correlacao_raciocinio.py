#!/usr/bin/env python3
"""Correlação entre o raciocínio PRODUZIDO e o acerto.

A medição anterior era degenerada: 100% das saídas tinham raciocínio, então
"com raciocínio vs sem raciocínio" não variava e a correlação dava 0.

O que varia de verdade é a QUALIDADE do traço. A pergunta científica é:

  quando o modelo lista o lance correto entre os candidatos, ele acerta mais?

E também: o traço longo ajuda ou atrapalha? Ele compara de fato, ou só descreve
um lance e o resto é enfeite?

Medidas:
  - P(acerto | solução entre os candidatos) vs P(acerto | solução ausente)
  - correlação ponto-bisserial entre "solução listada" e acerto
  - correlação entre número de candidatos e acerto
  - correlação entre comprimento do traço e acerto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr.dados import normalizar
from chess_rlvr.raciocinio import prompt_candidatos, prompt_forcados
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle

_RE_LANCE = re.compile(r"LANCE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
_RE_CAND = re.compile(r"^\s*[-*]\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.MULTILINE)


def parsear(texto: str) -> dict:
    texto = texto or ""
    candidatos = [c.lower() for c in _RE_CAND.findall(texto)]
    m = _RE_LANCE.search(texto)
    escolhido = m.group(1).lower() if m else None
    return {
        "candidatos": candidatos,
        "n_candidatos": len(candidatos),
        "escolhido": escolhido,
        "tem_comparacao": "compara" in texto.lower(),
        "tem_situacao": "situa" in texto.lower(),
        "palavras": len(texto.split()),
        "texto": texto,
    }


def ponto_bisserial(xs: list[int], ys: list[int]) -> float:
    """r entre uma variável binária (x) e uma binária (y)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="out/grpo-cot")
    ap.add_argument("--puzzles", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--formato", choices=["candidatos", "forcados"],
                    default="candidatos")
    ap.add_argument("--saida", default="out/correlacao-raciocinio.json")
    args = ap.parse_args()

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    itens = []
    for bruto in ds:
        if len(itens) >= args.puzzles:
            break
        l = normalizar(bruto)
        try:
            itens.append(montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"],
                                       moves=l["moves"], rating=l["rating"],
                                       temas=l["temas"], game_id=l["game_id"]))
        except PuzzleInvalido:
            continue
    print("puzzles:", len(itens), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.modelo, dtype="bfloat16", device_map="cuda:0"),
        args.adapter)
    modelo.eval()

    registros = []
    for i, p in enumerate(itens):
        prompt = (prompt_forcados(p.tabuleiro) if args.formato == "forcados"
                  else prompt_candidatos(p.tabuleiro))
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      return_tensors="pt", add_generation_prompt=True)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to("cuda")
        with torch.no_grad():
            out = modelo.generate(ids, max_new_tokens=args.max_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        texto = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        d = parsear(texto)
        d.pop("texto")
        alvo = p.primeiro_lance
        d["alvo"] = alvo
        d["acertou"] = int(d["escolhido"] == alvo)
        d["solucao_listada"] = int(alvo in d["candidatos"])
        d["escolheu_listado"] = int(d["escolhido"] is not None
                                    and d["escolhido"] in d["candidatos"])
        d["rating"] = p.rating
        registros.append(d)
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(itens)), flush=True)

    n = len(registros)
    acertos = [r["acertou"] for r in registros]
    listada = [r["solucao_listada"] for r in registros]
    ncand = [r["n_candidatos"] for r in registros]
    palavras = [r["palavras"] for r in registros]

    def taxa(mascara):
        sel = [r["acertou"] for r in registros if mascara(r)]
        return (sum(sel), len(sel), 100 * sum(sel) / max(1, len(sel)))

    a_com = taxa(lambda r: r["solucao_listada"] == 1)
    a_sem = taxa(lambda r: r["solucao_listada"] == 0)

    r_listada = ponto_bisserial(listada, acertos)
    r_ncand = pearson(ncand, acertos)
    r_palavras = pearson(palavras, acertos)

    # correlação condicionada: entre os que listam a solução, escolher importa?
    sub = [r for r in registros if r["solucao_listada"] == 1]
    r_escolha = ponto_bisserial([r["escolheu_listado"] for r in sub],
                                [r["acertou"] for r in sub]) if len(sub) > 2 else 0.0

    resumo = {
        "n": n,
        "acuracia_geral": 100 * sum(acertos) / n,
        "com_raciocinio_estruturado": 100 * sum(
            1 for r in registros if r["tem_situacao"] and r["tem_comparacao"]) / n,
        "solucao_listada": {
            "casos": sum(listada), "taxa": 100 * sum(listada) / n,
            "acerto_quando_listada": a_com[2], "n_quando_listada": a_com[1],
            "acerto_quando_ausente": a_sem[2], "n_quando_ausente": a_sem[1],
            "correlacao_ponto_bisserial": r_listada,
        },
        "candidatos_por_traco": sum(ncand) / n,
        "correlacao_n_candidatos": r_ncand,
        "palavras_por_traco": sum(palavras) / n,
        "correlacao_comprimento": r_palavras,
        "correlacao_escolher_entre_listados": r_escolha,
        "escolheu_um_dos_listados": 100 * sum(r["escolheu_listado"] for r in registros) / n,
    }

    print()
    print("=== CORRELACAO RACIOCINIO x ACERTO (n=%d) ===" % n)
    print("acuracia geral                  : %.2f%%" % resumo["acuracia_geral"])
    print("traco estruturado (situacao+comp): %.1f%%" % resumo["com_raciocinio_estruturado"])
    print()
    print("solucao entre os candidatos     : %d/%d = %.1f%%" % (
        sum(listada), n, 100 * sum(listada) / n))
    print("  acerto QUANDO listada         : %.2f%% (n=%d)" % (a_com[2], a_com[1]))
    print("  acerto QUANDO ausente         : %.2f%% (n=%d)" % (a_sem[2], a_sem[1]))
    print("  r ponto-bisserial             : %+.4f" % r_listada)
    print()
    print("candidatos por traco            : %.2f | r com acerto: %+.4f" % (
        resumo["candidatos_por_traco"], r_ncand))
    print("palavras por traco              : %.1f | r com acerto: %+.4f" % (
        resumo["palavras_por_traco"], r_palavras))
    print("escolheu um dos listados        : %.1f%%" % resumo["escolheu_um_dos_listados"])
    print("  r(escolher listado, acertar)  : %+.4f (n=%d)" % (r_escolha, len(sub)))

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(
        {"resumo": resumo, "itens": registros}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
