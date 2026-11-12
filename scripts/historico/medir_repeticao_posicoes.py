#!/usr/bin/env python3
"""Um n-grama de posicoes e sequer viavel em xadrez?

A ideia de "deixar a memorizacao no n-grama" so funciona se a mesma posicao
(or a mesma sequencia curta) reaparecer muitas vezes. Em xadrez o espaco de
estados e enorme (~10^44 posicoes legais), entao a pergunta empirica e:

  - quantas posicoes do TESTE aparecem no TREINO, exatamente?
  - quantos prefixos curtos de 2-4 lances se repetem?
  - um lookup por posicao cobriria que fracao do teste?

Se a resposta for ~0%, memorizar posicao nao cobre nada e a proposta de n-grama
para xadrez e incoerente — o que resta e o n-grama de MOTIVOS/padroes, que e
outra coisa.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess

from chess_rlvr.dados import normalizar
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle


def ler(caminho: str, limite: int | None):
    """Le um split jsonl e devolve (fens, chaves de padrao)."""
    fens, ngramas = [], []
    with open(caminho, encoding="utf-8") as fh:
        for i, linha in enumerate(fh):
            if limite and i >= limite:
                break
            try:
                d = json.loads(linha)
            except Exception:
                continue
            l = normalizar(d)
            try:
                p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                                  rating=l["rating"], temas=l["temas"], game_id=l["game_id"])
            except PuzzleInvalido:
                continue
            tab = p.tabuleiro
            # chave de posicao: FEN sem contadores (meia-jogada/numero do lance)
            campos = tab.fen().split()
            fens.append(" ".join(campos[:4]))
            # n-gramas do INICIO da posicao: quais pecas ocupam quais casas,
            # normalizado por cor a jogar — capta "mesma estrutura"
            pecas = sorted(
                (chess.square_name(c), ("w" if pc.color else "b") + str(pc.piece_type))
                for c, pc in tab.piece_map().items())
            base = "|".join("%s%s" % (c, p2) for c, p2 in pecas)
            ngramas.append(base)
    return fens, ngramas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", default="data/splits/treino.jsonl")
    ap.add_argument("--teste", default="data/splits/teste.jsonl")
    ap.add_argument("--limite-treino", type=int, default=200_000)
    ap.add_argument("--limite-teste", type=int, default=20_000)
    ap.add_argument("--saida", default="out/repeticao-posicoes.json")
    args = ap.parse_args()

    print("lendo treino...", flush=True)
    fens_tr, ng_tr = ler(args.treino, args.limite_treino)
    print("  treino:", len(fens_tr), "posicoes", flush=True)
    print("lendo teste...", flush=True)
    fens_te, ng_te = ler(args.teste, args.limite_teste)
    print("  teste:", len(fens_te), "posicoes", flush=True)

    set_fen = set(fens_tr)
    set_ng = set(ng_tr)

    exato = sum(1 for f in fens_te if f in set_fen)
    estrutural = sum(1 for g in ng_te if g in set_ng)

    c_tr = Counter(fens_tr)
    repeticoes_treino = sum(1 for _, n in c_tr.items() if n > 1)

    resumo = {
        "posicoes_treino": len(fens_tr),
        "posicoes_unicas_treino": len(set_fen),
        "posicoes_teste": len(fens_te),
        "posicoes_unicas_teste": len(set(fens_te)),
        "teste_cuja_posicao_exata_existe_no_treino_pct": 100.0 * exato / max(1, len(fens_te)),
        "teste_cuja_estrutura_existe_no_treino_pct": 100.0 * estrutural / max(1, len(fens_te)),
        "posicoes_repetidas_no_treino_pct": 100.0 * repeticoes_treino / max(1, len(set_fen)),
        "grau_de_repeticao_treino": len(fens_tr) / max(1, len(set_fen)),
    }

    print()
    print("=== POSICOES SE REPETEM? ===")
    print("posicoes unicas no treino       : %d de %d (%.2f%%)" % (
        resumo["posicoes_unicas_treino"], resumo["posicoes_treino"],
        100 * resumo["posicoes_unicas_treino"] / max(1, resumo["posicoes_treino"])))
    print("grau de repeticao (linhas/unica): %.3f" % resumo["grau_de_repeticao_treino"])
    print()
    print("=== COBERTURA DE UM LOOKUP POR POSICAO NO TESTE ===")
    print("posicao EXATA do teste esta no treino : %5.3f%%" % resumo["teste_cuja_posicao_exata_existe_no_treino_pct"])
    print("mesma ESTRUTURA de pecas no treino    : %5.3f%%" % resumo["teste_cuja_estrutura_existe_no_treino_pct"])

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resumo, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
