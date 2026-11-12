"""Verificação de puzzles de xadrez com máscara de legalidade e recompensa parcial.

Convenção do dataset Lichess (verificada empiricamente em `scripts/verificar_convencao.py`):
o FEN é a posição ANTES do lance do oponente; `Moves[0]` é esse lance do oponente.
Depois de aplicá-lo, o lado a jogar é o SOLVER, e seus lances são `Moves[1], Moves[3], ...`.

Este módulo é deliberadamente puro (sem I/O, sem rede) para ser testável em CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


class PuzzleInvalido(ValueError):
    """O registro não forma uma linha legal — deve ser quarentenado, não corrigido."""


@dataclass(frozen=True)
class Puzzle:
    id: str
    rating: int
    temas: tuple[str, ...]
    game_id: str
    # Posição já com o lance do oponente aplicado: é aqui que o solver joga.
    tabuleiro: chess.Board = field(compare=False)
    solucao: tuple[str, ...] = ()      # lances do solver, UCI
    respostas: tuple[str, ...] = ()    # respostas do oponente, UCI

    @property
    def lado(self) -> chess.Color:
        return self.tabuleiro.turn

    @property
    def primeiro_lance(self) -> str:
        if not self.solucao:
            raise PuzzleInvalido(f"{self.id}: solução vazia")
        return self.solucao[0]

    @property
    def fen(self) -> str:
        return self.tabuleiro.fen()


def montar_puzzle(
    *,
    puzzle_id: str,
    fen: str,
    moves: str,
    rating: int,
    temas: tuple[str, ...] = (),
    game_id: str = "",
) -> Puzzle:
    """Constrói um Puzzle a partir de uma linha do dataset Lichess.

    Levanta `PuzzleInvalido` quando a linha não é jogável. Não tenta consertar:
    registro ruim vira quarentena, nunca um alvo inventado.
    """
    ucis = moves.split()
    if len(ucis) < 2:
        raise PuzzleInvalido(f"{puzzle_id}: linha curta demais ({len(ucis)} lances)")

    tabuleiro = chess.Board(fen)

    # Moves[0] pertence ao lado a jogar no FEN (o oponente).
    try:
        lance_oponente = chess.Move.from_uci(ucis[0])
    except ValueError as exc:
        raise PuzzleInvalido(f"{puzzle_id}: UCI inválido em Moves[0]: {ucis[0]}") from exc
    if lance_oponente not in tabuleiro.legal_moves:
        raise PuzzleInvalido(f"{puzzle_id}: lance do oponente ilegal em {fen}: {ucis[0]}")
    tabuleiro.push(lance_oponente)

    solucao = tuple(ucis[1::2])
    respostas = tuple(ucis[2::2])
    if not solucao:
        raise PuzzleInvalido(f"{puzzle_id}: sem lance de solução")

    # Valida a linha inteira; qualquer ilegalidade invalida o registro.
    prova = tabuleiro.copy()
    for i, uci in enumerate(ucis[1:], start=1):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise PuzzleInvalido(f"{puzzle_id}: UCI inválido: {uci}") from exc
        if mv not in prova.legal_moves:
            raise PuzzleInvalido(f"{puzzle_id}: lance {i} ilegal ({uci}) em {prova.fen()}")
        prova.push(mv)

    return Puzzle(
        id=puzzle_id,
        rating=rating,
        temas=temas,
        game_id=game_id,
        tabuleiro=tabuleiro,
        solucao=solucao,
        respostas=respostas,
    )


# --------------------------------------------------------------------------
# Extração de lance da saída do modelo
# --------------------------------------------------------------------------

def normalizar_uci(texto: str) -> str | None:
    """Extrai o primeiro lance UCI plausível de um texto livre.

    Aceita `e2e4`, `e2-e4`, `e7e8q`, e ignora cercas de código e pontuação.
    Retorna None quando não há candidato — que é diferente de lance ilegal.
    """
    if not texto:
        return None
    import re

    # UCI canônico: casa origem, casa destino, promoção opcional.
    padrao = re.compile(r"\b([a-h][1-8])[- ]?([a-h][1-8])([qrbn])?\b", re.IGNORECASE)
    for m in padrao.finditer(texto):
        origem, destino, promo = m.group(1).lower(), m.group(2).lower(), (m.group(3) or "").lower()
        return f"{origem}{destino}{promo}"
    return None


# --------------------------------------------------------------------------
# Máscara de legalidade
# --------------------------------------------------------------------------

def lances_legais_uci(tabuleiro: chess.Board) -> set[str]:
    return {m.uci() for m in tabuleiro.legal_moves}


def mascarar_logits(logits_por_lance: dict[str, float], tabuleiro: chess.Board) -> dict[str, float]:
    """Zera (via -inf) qualquer lance ilegal.

    Usada para medir o efeito da máscara: sem ela o modelo gasta orçamento
    aprendendo que `Ke9` não existe.
    """
    legais = lances_legais_uci(tabuleiro)
    return {u: (v if u in legais else float("-inf")) for u, v in logits_por_lance.items()}


# --------------------------------------------------------------------------
# Recompensa
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultadoLance:
    status: str            # "correto" | "legal_bom" | "legal_ruim" | "ilegal" | "nao_parseavel"
    lance: str | None
    recompensa: float


PESOS = {
    "correto": 1.0,
    "legal_bom": 0.3,
    "legal_ruim": -0.2,
    "ilegal": -1.0,
    "nao_parseavel": -1.0,
}


def avaliar_lance(
    *,
    tabuleiro: chess.Board,
    lance_previsto: str | None,
    lance_solucao: str,
    avaliacao_alternativa: float | None = None,
) -> ResultadoLance:
    """Classifica um lance e devolve recompensa.

    `avaliacao_alternativa` é a avaliação (em centipeões, do ponto de vista de quem
    joga) do lance previsto, calculada por um oráculo externo (Stockfish). Quando
    ausente, um lance legal que não é o da solução recebe o peso conservador
    `legal_ruim`, nunca `legal_bom` por suposição.
    """
    if lance_previsto is None:
        return ResultadoLance("nao_parseavel", None, PESOS["nao_parseavel"])

    try:
        mv = chess.Move.from_uci(lance_previsto)
    except ValueError:
        return ResultadoLance("ilegal", lance_previsto, PESOS["ilegal"])

    if mv not in tabuleiro.legal_moves:
        return ResultadoLance("ilegal", lance_previsto, PESOS["ilegal"])

    if lance_previsto == lance_solucao:
        return ResultadoLance("correto", lance_previsto, PESOS["correto"])

    if avaliacao_alternativa is not None and avaliacao_alternativa >= 0:
        return ResultadoLance("legal_bom", lance_previsto, PESOS["legal_bom"])

    return ResultadoLance("legal_ruim", lance_previsto, PESOS["legal_ruim"])


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

PECAS_PT = {
    "P": "peão branco", "N": "cavalo branco", "B": "bispo branco",
    "R": "torre branca", "Q": "dama branca", "K": "rei branco",
    "p": "peão preto", "n": "cavalo preto", "b": "bispo preto",
    "r": "torre preta", "q": "dama preta", "k": "rei preto",
}


def descrever_posicao(tabuleiro: chess.Board) -> str:
    """Lista as peças por casa. É o formato que força rastreamento de estado:

    o modelo não pode 'sentir' o tabuleiro, precisa reconstruí-lo do texto.
    """
    linhas = []
    for casa, peca in sorted(tabuleiro.piece_map().items(), key=lambda kv: kv[0]):
        linhas.append(f"{chess.square_name(casa)}: {PECAS_PT[peca.symbol()]}")
    return "\n".join(linhas)


def montar_prompt(tabuleiro: chess.Board) -> str:
    lado = "brancas" if tabuleiro.turn == chess.WHITE else "pretas"
    return (
        "Você joga com as " + lado + " e é a sua vez.\n\n"
        "Posição (peças por casa):\n" + descrever_posicao(tabuleiro) + "\n\n"
        "FEN: " + tabuleiro.fen() + "\n\n"
        "Responda com o melhor lance em notação UCI (ex.: e2e4). "
        "Escreva apenas o lance."
    )
