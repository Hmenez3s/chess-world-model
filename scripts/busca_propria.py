#!/usr/bin/env python3
"""O modelo pensa à frente com o PRÓPRIO modelo de mundo? (sem treinar nada)

Mede três coisas nos mesmos puzzles do teste:

  1. guloso   — a saída normal (top5 → melhor), com lance verificado.
  2. pass@k   — com k amostras, a solução aparece em alguma? É o teto do que
                RL/STaR conseguem destravar (RLVR reorganiza a probabilidade
                dentro do pass@k; não cria o que não está lá).
  3. busca 2  — minimax de profundidade 2 usando só o modelo: para cada lance
                do top5 que ele mesmo escreveu, aplica o lance, pede ao modelo
                o top5 da posição do adversário e usa o melhor valor dele.
                valor(lance) = 100 − melhor valor do adversário.
                Nenhuma chamada ao Stockfish na inferência.

Se (3) superar (1), o modelo de mundo já serve para pensar à frente e o passo
seguinte é amortizar essa busca no treino (traços de busca + STaR).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avaliar_mav import carregar, wilson
from chess_rlvr import mav


def top5_de(texto: str) -> list[tuple[str, int]]:
    for ln in (texto or "").splitlines():
        if ln.startswith("top"):
            saida = []
            for item in ln.split(":", 1)[1].split():
                uci, _, valor = item.partition(":")
                try:
                    saida.append((uci.strip().lower(), int(valor)))
                except ValueError:
                    pass
            return saida
    return []


def gerar(modelo, tok, prompts, lote, max_tokens, amostrar=False, temperatura=0.8):
    textos = []
    for i in range(0, len(prompts), lote):
        bloco = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True)
                 for p in prompts[i:i + lote]]
        enc = tok(bloco, return_tensors="pt", padding=True).to("cuda")
        opcoes = {"max_new_tokens": max_tokens, "do_sample": amostrar,
                  "pad_token_id": tok.eos_token_id}
        if amostrar:
            opcoes["temperature"] = temperatura
        with torch.no_grad():
            out = modelo.generate(**enc, **opcoes)
        for j in range(len(bloco)):
            textos.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
        print("  gerados %d/%d" % (len(textos), len(prompts)), flush=True)
    return textos


def valor_para_quem_jogou(tab_depois: chess.Board, texto_adv: str) -> float:
    """Valor (0..100) do lance para quem o jogou, a partir da visão do adversário."""
    if tab_depois.is_checkmate():
        return 100.0
    if tab_depois.is_game_over(claim_draw=True):
        return 50.0
    legais = {m.uci() for m in tab_depois.legal_moves}
    valores = [v for u, v in top5_de(texto_adv) if u in legais]
    if not valores:
        return 50.0          # sem leitura do adversário: neutro, não inventa
    return 100.0 - max(valores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="data/splits/teste.jsonl")
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=440)
    ap.add_argument("--saida", default="out/busca-propria.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    itens = carregar(args.split, args.puzzles, args.seed)
    print("puzzles:", len(itens), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = AutoModelForCausalLM.from_pretrained(args.modelo, dtype=torch.bfloat16,
                                                  device_map="cuda:0")
    modelo = PeftModel.from_pretrained(modelo, args.adapter)
    modelo.eval()

    prompts = [mav.prompt_transicao(x["fen0"], x["m0"]) for x in itens]

    print("== 1. guloso ==", flush=True)
    gulosos = gerar(modelo, tok, prompts, args.lote, args.max_tokens)

    print("== 2. pass@%d ==" % args.k, flush=True)
    amostras = gerar(modelo, tok, [p for p in prompts for _ in range(args.k)],
                     args.lote, args.max_tokens, amostrar=True)

    print("== 3. busca de profundidade 2 com o proprio modelo ==", flush=True)
    candidatos = []          # (indice do puzzle, lance, tabuleiro depois)
    for i, (x, texto) in enumerate(zip(itens, gulosos)):
        legais = {m.uci() for m in x["tab"].legal_moves}
        vistos = []
        for u, _v in top5_de(texto):
            if u in legais and u not in vistos:
                vistos.append(u)
        if not vistos:
            lance, _ = mav.lance_verificado(texto, x["tab"])
            vistos = [lance] if lance else []
        for u in vistos:
            depois = x["tab"].copy()
            depois.push(chess.Move.from_uci(u))
            candidatos.append((i, u, depois))
    textos_adv = gerar(
        modelo, tok,
        [mav.prompt_posicao(d.fen()) for _i, _u, d in candidatos
         if not d.is_game_over(claim_draw=True)],
        args.lote, args.max_tokens)
    it_adv = iter(textos_adv)

    por_puzzle: dict[int, list[tuple[str, float]]] = {}
    for i, u, depois in candidatos:
        texto_adv = "" if depois.is_game_over(claim_draw=True) else next(it_adv)
        por_puzzle.setdefault(i, []).append((u, valor_para_quem_jogou(depois, texto_adv)))

    acertos = {"guloso": 0, "amostra_1": 0, "pass_k": 0, "maioria": 0, "busca_2": 0}
    registros = []
    for i, x in enumerate(itens):
        sol = x["solucao"]
        g, _ = mav.lance_verificado(gulosos[i], x["tab"])
        lances_k = [mav.lance_verificado(t, x["tab"])[0]
                    for t in amostras[i * args.k:(i + 1) * args.k]]
        maioria = Counter(l for l in lances_k if l).most_common(1)
        maioria = maioria[0][0] if maioria else None
        opcoes = por_puzzle.get(i, [])
        busca = max(opcoes, key=lambda par: par[1])[0] if opcoes else g
        acertos["guloso"] += g == sol
        acertos["amostra_1"] += lances_k[0] == sol
        acertos["pass_k"] += sol in lances_k
        acertos["maioria"] += maioria == sol
        acertos["busca_2"] += busca == sol
        registros.append({"rating": int(x["rating"]), "solucao": sol, "guloso": g,
                          "busca_2": busca, "candidatos": opcoes, "amostras": lances_k})

    n = len(itens)
    res = {"n": n, "k": args.k, "adapter": args.adapter, "itens": registros}
    print()
    print("=== O MODELO PENSA A FRENTE COM O PROPRIO MODELO DE MUNDO? (n=%d) ===" % n)
    for chave, rotulo in (("guloso", "guloso (saida normal)"),
                          ("amostra_1", "1 amostra (T=0,8)"),
                          ("maioria", "maioria de %d amostras" % args.k),
                          ("busca_2", "busca de profundidade 2"),
                          ("pass_k", "pass@%d (teto do RL/STaR)" % args.k)):
        k = acertos[chave]
        lo, hi = wilson(k, n)
        res[chave] = {"acertos": k, "pct": 100 * k / n, "ic95": [lo, hi]}
        print("%-30s %3d/%d = %5.2f%%  IC95 [%.2f; %.2f]" % (rotulo, k, n, 100 * k / n, lo, hi))

    a, b = acertos["guloso"], acertos["busca_2"]
    p = (a + b) / (2 * n)
    se = math.sqrt(p * (1 - p) * 2 / n) if 0 < p < 1 else 0
    res["z_busca_vs_guloso"] = (b - a) / n / se if se else 0.0
    print("busca vs guloso: z = %+.2f" % res["z_busca_vs_guloso"])

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
