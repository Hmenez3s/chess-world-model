#!/usr/bin/env python3
"""Escada de diagnóstico v2 — sondas balanceadas, com distribuição de respostas.

A v1 tinha três defeitos de medição que inflavam ou deflavam os números:

  1. percepção exigia "peão branco" quando "peão" já estava certo
  2. xeque é raro, então responder sempre "não" dava 98% — sonda desbalanceada
  3. legalidade não era 50/50

Esta versão corrige os três e reporta, para cada sonda:

  - acurácia
  - BALANÇO das classes (quantos sim / quantos não no gabarito)
  - DISTRIBUIÇÃO das respostas (se o modelo responde sempre a mesma coisa,
    a acurácia é artefato, não capacidade)

Sem a coluna de distribuição, "98% em xeque" parece ótimo e é apenas um
detector constante. Esse é o erro que a v1 cometeu.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr.dados import normalizar
from chess_rlvr.verificador import PuzzleInvalido, descrever_posicao, montar_puzzle

VALOR = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

NOME_PECA = {chess.PAWN: "peão", chess.KNIGHT: "cavalo", chess.BISHOP: "bispo",
             chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rei"}


def _sim_nao(texto: str):
    t = (texto or "").lower().strip()
    if re.match(r"^(sim|yes|s\b)", t):
        return True
    if re.match(r"^(não|nao|no|n\b)", t):
        return False
    # procura no meio, mas só a primeira ocorrência
    m = re.search(r"\b(sim|não|nao)\b", t)
    if m:
        return m.group(1) in ("sim",)
    return None


# ---------------------------------------------------------------------------
# Sondas
# ---------------------------------------------------------------------------

def sonda_percepcao(tab: chess.Board):
    """Tipo da peça numa casa. Aceita 'peão' — cor é outra sonda."""
    casas = sorted(tab.piece_map())
    casa = casas[len(casas) // 2]
    peca = tab.piece_at(casa)
    q = "Qual o tipo da peça na casa %s? Responda só o nome (rei, dama, torre, bispo, cavalo ou peão)." % chess.square_name(casa)
    esperado = NOME_PECA[peca.piece_type]
    def ok(r):
        t = (r or "").lower()
        return esperado in t
    return q, esperado, ok


def sonda_cor(tab: chess.Board):
    """Cor da peça numa casa. Separa percepção de tipo de percepção de cor."""
    casas = sorted(tab.piece_map())
    casa = casas[len(casas) // 2]
    peca = tab.piece_at(casa)
    q = "A peça na casa %s é das brancas ou das pretas? Responda só 'brancas' ou 'pretas'." % chess.square_name(casa)
    esperado = "brancas" if peca.color else "pretas"
    def ok(r):
        t = (r or "").lower()
        return esperado in t
    return q, esperado, ok


def sonda_contagem(tab: chess.Board):
    """Contar peões. Exige varredura da lista, não regra de xadrez."""
    n = sum(1 for p in tab.piece_map().values() if p.piece_type == chess.PAWN)
    q = "Quantos peões existem no tabuleiro no total (as duas cores)? Responda apenas o número."
    def ok(r):
        m = re.search(r"\d+", r or "")
        return bool(m) and int(m.group()) == n
    return q, str(n), ok


def sonda_material(tab: chess.Board):
    b = sum(VALOR[p.piece_type] for p in tab.piece_map().values() if p.color)
    p_ = sum(VALOR[p.piece_type] for p in tab.piece_map().values() if not p.color)
    esperado = "brancas" if b > p_ else ("pretas" if p_ > b else "equilibrado")
    q = ("Somando peão=1, cavalo=3, bispo=3, torre=5, dama=9, quem tem mais material? "
         "Responda 'brancas', 'pretas' ou 'equilibrado'.")
    def ok(r):
        t = (r or "").lower()
        if esperado == "equilibrado":
            return "equilibrad" in t
        return esperado in t
    return q, esperado, ok


def sonda_xeque(tab: chess.Board):
    """Detectar xeque. A distribuição de respostas é reportada à parte."""
    em = tab.is_check()
    q = "O rei do lado a quem cabe jogar está em xeque agora? Responda só 'sim' ou 'não'."
    def ok(r):
        return _sim_nao(r) is em
    return q, ("sim" if em else "não"), ok


def sonda_legalidade(tab: chess.Board, semente: int):
    """Julgar legalidade de um lance, balanceado 50/50 por construção."""
    legais = sorted(tab.legal_moves, key=lambda m: m.uci())
    # candidato ilegal: lance de cavalo geométrico que não é legal
    ilegais = []
    for de in chess.SQUARES:
        for d in (17, 15, 10, 6, -6, -10, -15, -17):
            para = de + d
            if 0 <= para < 64:
                mv = chess.Move(de, para)
                if mv not in tab.legal_moves:
                    ilegais.append(mv)
        if len(ilegais) > 40:
            break
    quer_legal = (semente % 2 == 0)
    if quer_legal and legais:
        mv, esperado = legais[semente % len(legais)], True
    elif ilegais:
        mv, esperado = ilegais[semente % len(ilegais)], False
    else:
        return None
    q = "O lance %s é legal nesta posição? Responda só 'sim' ou 'não'." % mv.uci()
    def ok(r):
        return _sim_nao(r) is esperado
    return q, ("sim" if esperado else "não"), ok


def sonda_lances_da_peca(tab: chess.Board):
    """Enumerar os lances legais de uma peça. A tarefa central de estado."""
    casas = [c for c, p in tab.piece_map().items() if p.color == tab.turn]
    if not casas:
        return None
    casa = casas[len(casas) // 2]
    certos = {m.uci() for m in tab.legal_moves if m.from_square == casa}
    if not certos:
        return None
    q = ("Liste TODOS os lances legais da peça que está na casa %s, em notação UCI, "
         "separados por vírgula. Responda apenas a lista." % chess.square_name(casa))
    def ok(r):
        achados = set(re.findall(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", (r or "").lower()))
        return achados == certos
    return q, sorted(certos), ok


SONDAS = [
    ("1_percepcao_tipo", sonda_percepcao),
    ("2_percepcao_cor", sonda_cor),
    ("3_contagem", sonda_contagem),
    ("4_material", sonda_material),
    ("5_xeque", sonda_xeque),
    ("6_legalidade", sonda_legalidade),
    ("7_lances_da_peca", sonda_lances_da_peca),
]

PROMPT = (
    "Você joga com as {lado} e é a sua vez.\n\n"
    "Posição (peças por casa):\n{posicao}\n\n"
    "FEN: {fen}\n\n"
    "{pergunta}"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--puzzles", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--saida", default="out/escada2.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    puzzles = []
    for bruto in ds:
        if len(puzzles) >= args.puzzles:
            break
        l = normalizar(bruto)
        try:
            puzzles.append(montar_puzzle(
                puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                rating=l["rating"], temas=l["temas"], game_id=l["game_id"]))
        except PuzzleInvalido:
            continue
    print("puzzles:", len(puzzles), flush=True)

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype="bfloat16", device_map="cuda:0")
    if args.adapter and Path(args.adapter).exists():
        modelo = PeftModel.from_pretrained(modelo, args.adapter)
    print("adapter:", args.adapter or "(nenhum)", flush=True)
    modelo.eval()

    resultados = {}
    for nome, construtor in SONDAS:
        acertos = tentativas = 0
        gabarito = Counter()
        respostas = Counter()
        exemplos_erro = []
        for i, p in enumerate(puzzles):
            item = (construtor(p.tabuleiro, i) if construtor is sonda_legalidade
                    else construtor(p.tabuleiro))
            if item is None:
                continue
            q, esperado, verificar = item
            lado = "brancas" if p.tabuleiro.turn else "pretas"
            prompt = PROMPT.format(lado=lado, posicao=descrever_posicao(p.tabuleiro),
                                   fen=p.tabuleiro.fen(), pergunta=q)
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          return_tensors="pt", add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            ids = ids.to("cuda")
            with torch.no_grad():
                out = modelo.generate(ids, max_new_tokens=args.max_tokens,
                                      do_sample=False, pad_token_id=tok.eos_token_id)
            texto = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            tentativas += 1
            ok = bool(verificar(texto))
            acertos += ok
            chave_esp = esperado if isinstance(esperado, str) else "conjunto"
            gabarito[chave_esp] += 1
            sn = _sim_nao(texto)
            respostas["sim" if sn is True else ("não" if sn is False else "outro")] += 1
            if not ok and len(exemplos_erro) < 3:
                exemplos_erro.append({"pergunta": q, "esperado": esperado,
                                      "resposta": texto[:80]})
        taxa = 100.0 * acertos / max(1, tentativas)
        resultados[nome] = {
            "n": tentativas, "acertos": acertos, "taxa": taxa,
            "gabarito": dict(gabarito), "respostas": dict(respostas),
            "exemplos_erro": exemplos_erro,
        }
        print("%-20s %4d/%4d = %6.2f%%  | gabarito %s | respostas %s" % (
            nome, acertos, tentativas, taxa, dict(gabarito), dict(respostas)), flush=True)

    print()
    print("=== ESCADA v2 (sondas balanceadas) ===")
    print("%-20s %-9s %-22s %s" % ("nivel", "taxa", "gabarito", "respostas"))
    for nome in resultados:
        r = resultados[nome]
        print("%-20s %6.2f%%   %-22s %s" % (nome, r["taxa"], r["gabarito"], r["respostas"]))

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(
        {"modelo": args.modelo, "adapter": args.adapter, "resultados": resultados},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)