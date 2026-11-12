from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.lig4 import (
    colunas_legais, gerar_posicoes, jogar, novo_tabuleiro, vencedor,
)


def test_vencedor_detecta_quatro_direcoes():
    direcoes = (
        ((5, 0), (5, 1), (5, 2), (5, 3)),  # horizontal
        ((2, 6), (3, 6), (4, 6), (5, 6)),  # vertical
        ((2, 0), (3, 1), (4, 2), (5, 3)),  # diagonal crescente
        ((2, 3), (3, 2), (4, 1), (5, 0)),  # diagonal decrescente
    )
    for casas in direcoes:
        tab = novo_tabuleiro()
        for r, c in casas:
            tab[r][c] = "X"
        assert vencedor(tab) == "X"


def test_posicoes_vencer_têm_lance_vencedor():
    for posicao in gerar_posicoes(10, "vencer", seed=42):
        assert vencedor(posicao["tabuleiro"]) is None
        assert posicao["corretas"]
        for coluna in posicao["corretas"]:
            tab = [linha[:] for linha in posicao["tabuleiro"]]
            jogar(tab, coluna, posicao["lado"])
            assert vencedor(tab) == posicao["lado"]


def test_posicoes_bloquear_têm_ameaça_unica_e_resposta_certa():
    for posicao in gerar_posicoes(10, "bloquear", seed=42):
        tab = posicao["tabuleiro"]
        lado = posicao["lado"]
        oponente = "O" if lado == "X" else "X"
        assert vencedor(tab) is None
        ameacas = set()
        for coluna in colunas_legais(tab):
            tentativa = [linha[:] for linha in tab]
            jogar(tentativa, coluna, oponente)
            if vencedor(tentativa) == oponente:
                ameacas.add(coluna)
        assert ameacas == posicao["corretas"]
        for coluna in posicao["corretas"]:
            bloqueado = [linha[:] for linha in tab]
            jogar(bloqueado, coluna, lado)
            for proxima in colunas_legais(bloqueado):
                resposta = [linha[:] for linha in bloqueado]
                jogar(resposta, proxima, oponente)
                assert vencedor(resposta) != oponente
