"""Testes do formato MAV: a grade tem de ser reconstruível a partir do texto."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import pytest

from chess_rlvr import mav

FEN = "r4rk1/2q3p1/p1bbp3/1p3p2/4NP2/5Q1P/PPPB2P1/4RRK1 w - - 0 20"


def test_grade_usa_maiuscula_para_branca():
    tab = chess.Board(FEN)
    linhas = mav.grade(tab).splitlines()
    assert len(linhas) == 9                       # 8 fileiras + rótulo de colunas
    assert linhas[-1].split() == list("abcdefgh")
    # fileira 4 = "4NP2": cavalo branco em e4, peão branco em f4
    fileira4 = linhas[4].split()
    assert fileira4[0] == "4"
    assert fileira4[1:] == list("....NP..")


def test_grade_tem_64_casas():
    grade = mav.grade(chess.Board(FEN))
    casas = [c for ln in grade.splitlines()[:8] for c in ln.split()[1:]]
    assert len(casas) == 64
    assert sum(1 for c in casas if c != ".") == len(chess.Board(FEN).piece_map())


def test_chance_vitoria_mapeia_extremos():
    assert mav.chance_vitoria(0) == 50
    assert mav.chance_vitoria(10000) == 100
    assert mav.chance_vitoria(-10000) == 0
    # monótona
    assert mav.chance_vitoria(300) > mav.chance_vitoria(100) > mav.chance_vitoria(0)


def test_resposta_e_reconstruivel():
    """O texto gerado tem de carregar a grade exata — é o que a avaliação confere."""
    tab = chess.Board(FEN)
    tab.push_uci("e4d6")
    texto = mav.resposta(tab, [("c6f3", 85), ("c7d6", 40)])
    campos = mav.extrair_campos(texto)
    assert campos["grade"] == mav.grade(tab)
    assert campos["melhor"] == "c6f3"
    assert campos["legais"] == {m.uci() for m in tab.legal_moves}


def test_extrair_campos_resiste_a_texto_vazio():
    campos = mav.extrair_campos("")
    assert campos["grade"] is None and campos["melhor"] is None


@pytest.mark.parametrize("uci", ["e4d6", "c6f3"])
def test_prompt_de_transicao_carrega_fen_e_lance(uci):
    texto = mav.prompt_transicao(FEN, uci)
    assert FEN in texto and uci in texto

def test_lance_verificado_prefere_o_lance_do_modelo():
    tab = chess.Board(FEN)
    tab.push_uci("e4d6")
    texto = mav.resposta(tab, [("c6f3", 85), ("c7d6", 40)])
    lance, origem = mav.lance_verificado(texto, tab)
    assert lance == "c6f3" and origem == "modelo"


def test_lance_verificado_cai_para_o_top5_quando_o_melhor_e_ilegal():
    tab = chess.Board(FEN)
    tab.push_uci("e4d6")
    # "melhor" ilegal de proposito; o top5 traz um lance valido
    texto = "top5: a1a8:90 c6f3:80\nmelhor: a1a8\nlegais: c6f3 c7d6"
    lance, origem = mav.lance_verificado(texto, tab)
    assert lance == "c6f3" and origem == "top5"


def test_lance_verificado_cai_para_a_lista_de_legais():
    tab = chess.Board(FEN)
    tab.push_uci("e4d6")
    texto = "top5: a1a8:90\nmelhor: a1a8\nlegais: c6f3 c7d6"
    lance, origem = mav.lance_verificado(texto, tab)
    assert lance in {"c6f3", "c7d6"} and origem == "legais"


def test_lance_verificado_sem_candidato_valido():
    tab = chess.Board(FEN)
    tab.push_uci("e4d6")
    texto = "top5: a1a8:90\nmelhor: a1a8\nlegais: a1a8"
    lance, origem = mav.lance_verificado(texto, tab)
    assert lance is None and origem == "nenhum"
