"""Formato de treino de "modelo de mundo" do jogo, adaptado do MAV.

Referência: Schultz et al., "Mastering Board Games by External and Internal
Planning with Language Models" (DeepMind, arXiv 2412.12119, v3 2025).
Um transformer de 1B treinado assim chega a ~2.820 Elo estimado sem busca.

O que o MAV ensina e o nosso treino anterior não ensinava:
  1. transição de estado: posição anterior + lance -> tabuleiro resultante
  2. lances legais da posição
  3. valor de cada lance (top-k ordenado com chance de vitória)
  4. melhor lance

Nós treinávamos só o item 4 ("behavioral cloning"), que é o alvo mais fraco:
no ChessBench (arXiv 2402.04494) valor-por-lance acerta 83,3% dos puzzles,
contra 65,7% de prever só o melhor lance, no mesmo modelo.

O tabuleiro sai como grade 8x8, um caractere por casa (maiúscula = branca,
minúscula = preta), com rótulos de fileira e coluna. Cada casa vira um token
em posição fixa, como no MAV, em vez de depender de como o BPE quebra o FEN.
"""

from __future__ import annotations

import math

import chess

TOP_K = 5


def grade(tab: chess.Board) -> str:
    linhas = []
    for fileira in range(7, -1, -1):
        casas = []
        for coluna in range(8):
            peca = tab.piece_at(chess.square(coluna, fileira))
            casas.append(peca.symbol() if peca else ".")
        linhas.append("%d %s" % (fileira + 1, " ".join(casas)))
    linhas.append("  a b c d e f g h")
    return "\n".join(linhas)


def chance_vitoria(cp: int) -> int:
    """Centipeões -> chance de vitória 0..100 (fórmula do Lichess)."""
    p = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
    return max(0, min(100, int(round(p))))


def _lado(tab: chess.Board) -> str:
    return "brancas" if tab.turn == chess.WHITE else "pretas"


def prompt_transicao(fen_anterior: str, lance: str) -> str:
    return ("FEN anterior: %s\nLance jogado: %s\n"
            "Mostre o tabuleiro resultante, os lances legais, os %d melhores "
            "com chance de vitória e o melhor lance." % (fen_anterior, lance, TOP_K))


def prompt_posicao(fen: str) -> str:
    return ("FEN: %s\nMostre o tabuleiro, os lances legais, os %d melhores "
            "com chance de vitória e o melhor lance." % (fen, TOP_K))


def resposta(tab: chess.Board, top: list[tuple[str, int]]) -> str:
    """top: [(uci, chance 0..100)] ordenado do melhor para o pior."""
    legais = " ".join(sorted(m.uci() for m in tab.legal_moves))
    linhas = [
        "estado:",
        grade(tab),
        "vez: %s" % _lado(tab),
        "legais: %s" % legais,
        "top%d: %s" % (TOP_K, " ".join("%s:%d" % (m, v) for m, v in top)),
        "melhor: %s" % top[0][0],
    ]
    return "\n".join(linhas)


def avaliar_top(tab: chess.Board, motor, profundidade: int = 10) -> list[tuple[str, int]]:
    """Top-k lances com chance de vitória, do ponto de vista de quem joga."""
    infos = motor.analyse(tab, chess.engine.Limit(depth=profundidade),
                          multipv=min(TOP_K, tab.legal_moves.count()))
    if isinstance(infos, dict):
        infos = [infos]
    top = []
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        cp = info["score"].pov(tab.turn).score(mate_score=10000)
        top.append((info["pv"][0].uci(), chance_vitoria(cp)))
    return top


# --------------------------------------------------------------------------
# Leitura da resposta (avaliação)
# --------------------------------------------------------------------------

def extrair_campos(texto: str) -> dict:
    campos: dict = {"grade": None, "legais": None, "melhor": None}
    linhas = (texto or "").splitlines()
    for i, ln in enumerate(linhas):
        if ln.strip() == "estado:" and i + 9 <= len(linhas):
            campos["grade"] = "\n".join(linhas[i + 1:i + 10])
        elif ln.startswith("legais:"):
            campos["legais"] = set(ln.split(":", 1)[1].split())
        elif ln.startswith("melhor:"):
            v = ln.split(":", 1)[1].strip().split()
            campos["melhor"] = v[0] if v else None
    return campos


def lance_verificado(texto: str, tabuleiro: chess.Board) -> tuple[str | None, str]:
    """Lance final conferido contra a posição real.

    O modelo escreve a própria lista de lances legais e um top-5 com valores.
    Quando o `melhor:` sai ilegal, a saída dele ainda contém candidatos válidos:
    usa o primeiro lance do `top5:` que seja legal; se nenhum for, o primeiro de
    `legais:` que seja legal.

    Devolve (lance, origem), com origem em {"modelo", "top5", "legais", "nenhum"}.
    Isso não dá informação nova ao modelo — só impede que ele jogue fora da
    própria lista.
    """
    campos = extrair_campos(texto)
    legais_reais = {m.uci() for m in tabuleiro.legal_moves}

    melhor = campos["melhor"]
    if melhor in legais_reais:
        return melhor, "modelo"

    for linha in (texto or "").splitlines():
        if not linha.startswith("top"):
            continue
        for item in linha.split(":", 1)[1].split():
            uci = item.split(":")[0].strip().lower()
            if uci in legais_reais:
                return uci, "top5"

    for uci in sorted(campos["legais"] or ()):
        if uci in legais_reais:
            return uci, "legais"

    return None, "nenhum"
