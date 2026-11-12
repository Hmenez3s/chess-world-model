"""Testes das invariantes que o README afirma.

Cobrem o que pode quebrar silenciosamente e invalidar resultados:
  - a máscara de legalidade realmente barra lance ilegal
  - o espelho é isomorfismo (mesma tática, linha legal)
  - a divisão por GameId não vaza
  - o IC de Wilson tem as propriedades de um IC

Não testam o modelo nem as GPUs: isso é medido pelos scripts em scripts/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import pytest

from chess_rlvr.avaliacao import wilson
from chess_rlvr.dados import dividir_por_jogo, auditar_vazamento, faixa_de_rating
from chess_rlvr.espelho import espelhar_tabuleiro, espelhar_uci, espelhar_puzzle
from chess_rlvr.verificador import (
    avaliar_lance, lances_legais_uci, mascarar_logits, montar_puzzle,
    normalizar_uci,
)

# Posição inicial e uma posição de mate em 1 bem conhecida.
INICIAL = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_EM_1 = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K w - - 1 25"  # Qh6-h7#


# --------------------------------------------------------------------------
# verificador
# --------------------------------------------------------------------------

def test_normalizar_uci_aceita_formatos_comuns():
    assert normalizar_uci("e2e4") == "e2e4"
    assert normalizar_uci("  e2-e4 ") == "e2e4"
    assert normalizar_uci("E2E4") == "e2e4"
    assert normalizar_uci("e7e8q") == "e7e8q"


def test_normalizar_uci_rejeita_lixo():
    assert normalizar_uci("") is None
    assert normalizar_uci("nada") is None
    assert normalizar_uci("z9z9") is None


def test_mascara_barra_lance_ilegal():
    tab = chess.Board(INICIAL)
    logits = {"e2e4": 5.0, "d2d4": 4.0, "e2e5": 99.0, "a1a8": 88.0}
    mascarado = mascarar_logits(logits, tab)
    # os ilegais precisam sumir ou virar -inf
    for ilegal in ("e2e5", "a1a8"):
        assert ilegal not in mascarado or mascarado[ilegal] == float("-inf")
    # os legais permanecem
    assert mascarado.get("e2e4") == 5.0


def test_mascara_nunca_deixa_ilegal_passar():
    tab = chess.Board(MATE_EM_1)
    legais = lances_legais_uci(tab)
    assert "h6h7" in legais
    # um lance geometricamente possível mas ilegal aqui
    assert "h6h8" not in legais


def test_avaliar_lance_acerta_o_mate():
    tab = chess.Board(MATE_EM_1)
    r = avaliar_lance(tabuleiro=tab, lance_previsto="h6h7", lance_solucao="h6h7")
    assert r.status == "correto"
    assert r.recompensa == 1.0


def test_avaliar_lance_ilegal_recebe_peso_negativo():
    tab = chess.Board(MATE_EM_1)
    r = avaliar_lance(tabuleiro=tab, lance_previsto="h6h8", lance_solucao="h6h7")
    assert r.status == "ilegal"
    assert r.recompensa < 0


def test_avaliar_lance_legal_errado_nao_vira_bom_por_suposicao():
    """Sem oráculo, um lance legal que não é a solução NÃO pode ser premiado."""
    tab = chess.Board(INICIAL)
    r = avaliar_lance(tabuleiro=tab, lance_previsto="e2e4", lance_solucao="d2d4")
    assert r.status == "legal_ruim"
    assert r.recompensa < 0


def test_montar_puzzle_comeca_no_lance_do_solver():
    # Moves[0] é do oponente; o solver joga Moves[1].
    p = montar_puzzle(puzzle_id="teste", fen=INICIAL, moves="e2e4 e7e5",
                      rating=1500, temas=("opening",), game_id="g1")
    assert p.primeiro_lance == "e7e5"
    assert p.lado == chess.BLACK
    assert p.tabuleiro.turn == chess.BLACK


# --------------------------------------------------------------------------
# espelho
# --------------------------------------------------------------------------

def test_espelhar_casa_e_involucao():
    from chess_rlvr.espelho import espelhar_casa
    for casa in chess.SQUARES:
        assert espelhar_casa(espelhar_casa(casa)) == casa


def test_espelhar_uci_preserva_promocao():
    assert espelhar_uci("e2e4") == "e7e5"
    assert espelhar_uci("e7e8q") == "e2e1q"
    assert espelhar_uci("a1h8") == "a8h1"


def test_espelho_preserva_legalidade():
    """Se o espelho não preserva legalidade, o teste de memorização é inválido."""
    for fen in (INICIAL, MATE_EM_1):
        tab = chess.Board(fen)
        esp = espelhar_tabuleiro(tab)
        # mesma contagem de peças, lados trocados
        assert len(tab.piece_map()) == len(esp.piece_map())
        assert esp.turn != tab.turn
        # cada lance legal vira um lance legal
        for mv in tab.legal_moves:
            uci = mv.uci()
            assert chess.Move.from_uci(espelhar_uci(uci)) in esp.legal_moves


def test_espelhar_puzzle_mantem_rating_e_inverte_a_solucao():
    p = montar_puzzle(puzzle_id="x", fen=INICIAL, moves="e2e4 e7e5 d2d4",
                      rating=1500, temas=("opening",), game_id="g1")
    e = espelhar_puzzle(p)
    assert e.rating == p.rating
    assert e.solucao == tuple(espelhar_uci(u) for u in p.solucao)
    # a linha espelhada é legal do começo ao fim
    tab = chess.Board(e.tabuleiro.fen())
    for uci in e.solucao:
        mv = chess.Move.from_uci(uci)
        assert mv in tab.legal_moves
        tab.push(mv)


# --------------------------------------------------------------------------
# dados: vazamento
# --------------------------------------------------------------------------

def _linha(i: int, game: str) -> dict:
    return {"puzzle_id": "p%d" % i, "game_id": game, "fen": INICIAL,
            "moves": "e2e4 e7e5", "rating": 1200 + i,
            "temas": ["opening"], "popularidade": 95}


def test_divisao_por_jogo_nao_vaza():
    """Puzzles da mesma partida precisam cair na mesma divisão."""
    linhas = [_linha(i, "g%d" % (i // 5)) for i in range(200)]
    div = dividir_por_jogo(linhas, seed=7)
    rel = auditar_vazamento(div)
    assert len(rel["jogos_compartilhados"]) == 0
    assert len(rel["puzzles_compartilhados"]) == 0
    # todo puzzle foi para exatamente uma divisão
    total = sum(len(getattr(div, n)) for n in ("treino", "validacao", "teste"))
    assert total == len(linhas)


def test_divisao_e_deterministica_por_seed():
    linhas = [_linha(i, "g%d" % (i // 5)) for i in range(200)]
    a = dividir_por_jogo(linhas, seed=7)
    b = dividir_por_jogo(linhas, seed=7)
    assert [x["puzzle_id"] for x in a.treino] == [x["puzzle_id"] for x in b.treino]
    c = dividir_por_jogo(linhas, seed=8)
    assert [x["puzzle_id"] for x in a.treino] != [x["puzzle_id"] for x in c.treino]


def test_faixas_de_rating_sao_contiguas():
    assert faixa_de_rating(0) == faixa_de_rating(599)
    assert faixa_de_rating(600) != faixa_de_rating(599)
    # a última faixa é aberta, sem teto
    assert faixa_de_rating(2600) == "2600-3000"
    assert faixa_de_rating(3000) == faixa_de_rating(5000) == "3000+"


# --------------------------------------------------------------------------
# avaliacao: IC de Wilson
# --------------------------------------------------------------------------

def test_wilson_tem_propriedades_de_ic():
    lo, hi = wilson(5, 100)
    assert 0.0 <= lo < 0.05 < hi <= 1.0
    # mais dados -> intervalo mais estreito
    lo2, hi2 = wilson(50, 1000)
    assert (hi2 - lo2) < (hi - lo)
    # 0 sucessos ainda tem limite superior positivo
    lo0, hi0 = wilson(0, 600)
    assert lo0 == 0.0 and hi0 > 0.0
    # 100% de sucesso tem limite inferior abaixo de 1
    lo100, hi100 = wilson(600, 600)
    assert lo100 > 0.0 and hi100 == pytest.approx(1.0, abs=1e-9)
