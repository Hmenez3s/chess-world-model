"""Motor simples de Lig 4 para gerar posições de avaliação."""

from __future__ import annotations

import random

LINHAS = 6
COLUNAS = 7
VAZIO = "."


def novo_tabuleiro() -> list[list[str]]:
    """Cria uma matriz 6x7 com as linhas em ordem visual (topo primeiro)."""
    return [[VAZIO for _ in range(COLUNAS)] for _ in range(LINHAS)]


def colunas_legais(tab: list[list[str]]) -> list[int]:
    """Retorna as colunas jogáveis, numeradas de 1 a 7."""
    return [c + 1 for c in range(COLUNAS) if tab[0][c] == VAZIO]


def jogar(tab: list[list[str]], coluna: int, lado: str) -> None:
    """Solta ``lado`` na coluna (1–7), modificando o tabuleiro in-place."""
    if lado not in ("X", "O"):
        raise ValueError("lado deve ser 'X' ou 'O'")
    if not 1 <= coluna <= COLUNAS:
        raise ValueError("coluna deve estar entre 1 e 7")
    c = coluna - 1
    for r in range(LINHAS - 1, -1, -1):
        if tab[r][c] == VAZIO:
            tab[r][c] = lado
            return
    raise ValueError("coluna cheia")


def vencedor(tab: list[list[str]]) -> str | None:
    """Retorna ``X``/``O`` se houver quatro em linha, caso contrário None."""
    for r in range(LINHAS):
        for c in range(COLUNAS):
            lado = tab[r][c]
            if lado == VAZIO:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + 3 * dr, c + 3 * dc
                if 0 <= er < LINHAS and 0 <= ec < COLUNAS and all(
                    tab[r + i * dr][c + i * dc] == lado for i in range(1, 4)
                ):
                    return lado
    return None


def _jogadas_vencedoras(tab: list[list[str]], lado: str) -> set[int]:
    vencedoras = set()
    for coluna in colunas_legais(tab):
        copia = [linha[:] for linha in tab]
        jogar(copia, coluna, lado)
        if vencedor(copia) == lado:
            vencedoras.add(coluna)
    return vencedoras


def gerar_posicoes(n: int, tipo: str, seed: int) -> list[dict]:
    """Gera ``n`` posições distintas apropriadas para ``vencer`` ou ``bloquear``.

    Cada item contém ``tabuleiro``, ``lado`` e ``corretas`` (colunas 1–7).
    Os tabuleiros são matrizes mutáveis; cada posição recebe sua própria cópia.
    """
    if n < 0:
        raise ValueError("n não pode ser negativo")
    if tipo not in ("vencer", "bloquear"):
        raise ValueError("tipo deve ser 'vencer' ou 'bloquear'")
    if n == 0:
        return []

    rng = random.Random(seed)
    encontradas: dict[tuple, dict] = {}
    tentativas = 0
    while len(encontradas) < n and tentativas < max(100_000, n * 2_000):
        tentativas += 1
        tab = novo_tabuleiro()
        lado = "X"
        while colunas_legais(tab) and vencedor(tab) is None:
            legais = colunas_legais(tab)
            if tipo == "vencer":
                corretas = _jogadas_vencedoras(tab, lado)
                if corretas:
                    chave = (tuple("".join(l) for l in tab), lado)
                    encontradas.setdefault(chave, {
                        "tabuleiro": [linha[:] for linha in tab],
                        "lado": lado,
                        "corretas": set(corretas),
                    })
                    break
            else:
                if not _jogadas_vencedoras(tab, lado):
                    ameacas = _jogadas_vencedoras(tab, "O" if lado == "X" else "X")
                    if len(ameacas) == 1:
                        chave = (tuple("".join(l) for l in tab), lado)
                        encontradas.setdefault(chave, {
                            "tabuleiro": [linha[:] for linha in tab],
                            "lado": lado,
                            "corretas": set(ameacas),
                        })
                        break
            jogar(tab, rng.choice(legais), lado)
            if vencedor(tab) is not None:
                break
            lado = "O" if lado == "X" else "X"

    if len(encontradas) < n:
        raise RuntimeError(
            "não foi possível gerar %d posições distintas do tipo %r (geradas %d)"
            % (n, tipo, len(encontradas))
        )
    return list(encontradas.values())[:n]


def desenhar(tab: list[list[str]]) -> str:
    """Renderiza o tabuleiro de cima para baixo, com colunas numeradas."""
    return "\n".join([" ".join(linha) for linha in tab] + ["1 2 3 4 5 6 7"])
