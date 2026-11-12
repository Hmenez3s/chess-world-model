"""Avaliação de lances com Stockfish, para recompensa parcial.

Sem isto, a recompensa é binária (acertou/errou) e ~90% dos grupos do GRPO ficam
com variância zero — o modelo não aprende. Stockfish dá um gradiente contínuo:
"não é o lance da solução, mas também não estraga a posição".

O motor é usado em profundidade baixa e determinística para ser rápido e reprodutível.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine

PROFUNDIDADE_PADRAO = 8


def localizar_motor() -> str | None:
    """Acha o binário do Stockfish sem depender de sudo ou PATH global."""
    candidatos = [
        os.environ.get("STOCKFISH_PATH"),
        os.path.expanduser("~/bin/stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        shutil.which("stockfish"),
    ]
    for c in candidatos:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@dataclass
class Motor:
    """Wrapper fino sobre python-chess.engine, com fechamento garantido."""
    caminho: str
    profundidade: int = PROFUNDIDADE_PADRAO
    _engine: object = None

    def abrir(self):
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.caminho)
        return self._engine

    def fechar(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, *exc):
        self.fechar()
        return False

    def avaliar_apos(self, tabuleiro: chess.Board, lance: chess.Move) -> float:
        """Avaliação em centipeões do ponto de vista de QUEM JOGA em `tabuleiro`.

        Positivo = bom para quem tem o lance. Escala Mate como ±10000.
        """
        copia = tabuleiro.copy()
        copia.push(lance)
        info = self.abrir().analyse(copia, chess.engine.Limit(depth=self.profundidade))
        score = info["score"].pov(tabuleiro.turn)
        # `score(mate_score=...)` ja converte mate para o valor sentinela com o
        # sinal correto. Nao tratar mate a mao: mate em 0 (xeque-mate ja dado)
        # tem `mate() == 0` e cairia no ramo negativo.
        return float(score.score(mate_score=10000))


def normalizar_vantagem(centipeoes: float, escala: float = 300.0) -> float:
    """Comprime a avaliação para [-1, 1], saturando em ~3 peões.

    Evita que uma posição já ganha domine a recompensa: a diferença entre +8 e +12
    peões não interessa ao aprendizado, mas a diferença entre 0 e +1 interessa muito.
    """
    if escala <= 0:
        return 0.0
    x = centipeoes / escala
    return max(-1.0, min(1.0, x))
