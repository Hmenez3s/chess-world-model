"""Espelhamento de posições como teste de memorização.

Ideia central: a tática de um puzzle é invariante sob reflexão vertical do tabuleiro
com troca de cores. Um modelo que *entende* a posição acerta a versão espelhada com a
mesma taxa. Um modelo que *decorou* o puzzle cai para perto de zero, porque as
coordenadas mudaram.

Isso transforma "ele memorizou?" de opinião em medida.
"""

from __future__ import annotations

import chess


def espelhar_casa(casa: int) -> int:
    """Reflete a casa verticalmente: a1 <-> a8, e2 <-> e7."""
    file_ = chess.square_file(casa)
    rank = chess.square_rank(casa)
    return chess.square(file_, 7 - rank)


def espelhar_uci(uci: str) -> str:
    mv = chess.Move.from_uci(uci)
    promo = mv.promotion
    return chess.Move(
        espelhar_casa(mv.from_square),
        espelhar_casa(mv.to_square),
        promotion=promo,
    ).uci()


def espelhar_tabuleiro(tabuleiro: chess.Board) -> chess.Board:
    """Reflexão vertical + troca de cores.

    É um isomorfismo de xadrez: a posição resultante é legal e a mesma tática vale.
    Direitos de roque são trocados de lado junto com as cores; en passant é refletido.
    """
    novo = chess.Board(None)  # tabuleiro vazio

    for casa, peca in tabuleiro.piece_map().items():
        nova_casa = espelhar_casa(casa)
        nova_peca = chess.Piece(peca.piece_type, not peca.color)
        novo.set_piece_at(nova_casa, nova_peca)

    # Turno invertido: quem jogava continua a jogar, mas com a outra cor.
    novo.turn = not tabuleiro.turn

    # Roque: o direito de um lado vira o direito do outro lado na cor oposta.
    if tabuleiro.has_kingside_castling_rights(chess.WHITE):
        novo.castling_rights |= chess.BB_H8
    if tabuleiro.has_queenside_castling_rights(chess.WHITE):
        novo.castling_rights |= chess.BB_A8
    if tabuleiro.has_kingside_castling_rights(chess.BLACK):
        novo.castling_rights |= chess.BB_H1
    if tabuleiro.has_queenside_castling_rights(chess.BLACK):
        novo.castling_rights |= chess.BB_A1

    if tabuleiro.ep_square is not None:
        novo.ep_square = espelhar_casa(tabuleiro.ep_square)

    novo.halfmove_clock = tabuleiro.halfmove_clock
    novo.fullmove_number = tabuleiro.fullmove_number

    return novo


class EspelhoInvalido(ValueError):
    """A posição espelhada não é jogável — o puzzle não serve para o teste."""


def espelhar_puzzle(puzzle):
    """Devolve uma cópia espelhada do puzzle, com a solução transformada.

    Levanta `EspelhoInvalido` quando a posição refletida não é legal (por exemplo,
    peões em fileiras terminais). Esses casos são excluídos da métrica, não contados
    como falha do modelo.
    """
    novo_tab = espelhar_tabuleiro(puzzle.tabuleiro)

    if novo_tab.status() != chess.STATUS_VALID:
        raise EspelhoInvalido(f"{puzzle.id}: posição espelhada inválida ({novo_tab.status()})")

    solucao = tuple(espelhar_uci(u) for u in puzzle.solucao)
    respostas = tuple(espelhar_uci(u) for u in puzzle.respostas)

    # Prova de que o espelho é fiel: a linha transformada tem de ser legal.
    prova = novo_tab.copy()
    for i, uci in enumerate(solucao):
        mv = chess.Move.from_uci(uci)
        if mv not in prova.legal_moves:
            raise EspelhoInvalido(
                f"{puzzle.id}: solução espelhada ilegal no lance {i} ({uci})"
            )
        prova.push(mv)
        if i < len(respostas):
            mv_r = chess.Move.from_uci(respostas[i])
            if mv_r not in prova.legal_moves:
                raise EspelhoInvalido(
                    f"{puzzle.id}: resposta espelhada ilegal no lance {i} ({respostas[i]})"
                )
            prova.push(mv_r)

    from .verificador import Puzzle

    return Puzzle(
        id=puzzle.id + "::espelho",
        rating=puzzle.rating,
        temas=puzzle.temas,
        game_id=puzzle.game_id,
        tabuleiro=novo_tab,
        solucao=solucao,
        respostas=respostas,
    )
