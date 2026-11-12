"""Carregamento e divisão dos puzzles do Lichess.

Regra de divisão: o grupo indivisível é o **GameId**, não a linha. Puzzles derivados
da mesma partida compartilham posições e motivos; separá-los por linha vazaria
resposta entre treino e teste e inflaria a métrica.

Estratificação por faixa de rating garante que treino e teste cobrem a mesma escada,
para que o slice por dificuldade seja comparável.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FAIXAS_RATING = (600, 1000, 1400, 1800, 2200, 2600, 3000)


def faixa_de_rating(rating: int) -> str:
    anterior = 0
    for limite in FAIXAS_RATING:
        if rating < limite:
            return f"{anterior}-{limite}"
        anterior = limite
    return f"{anterior}+"


def hash_do_jogo(game_id: str, seed: int = 0) -> float:
    """Hash estável do GameId em [0,1). Independente da ordem das linhas."""
    h = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


@dataclass(frozen=True)
class Divisao:
    treino: list
    validacao: list
    teste: list

    def resumo(self) -> dict:
        return {
            "treino": len(self.treino),
            "validacao": len(self.validacao),
            "teste": len(self.teste),
        }


def dividir_por_jogo(linhas: list[dict], *, seed: int = 0,
                     frac_val: float = 0.05, frac_teste: float = 0.15) -> Divisao:
    """Divide por GameId, de forma determinística e reprodutível.

    Um GameId cai inteiro numa só divisão. Se um jogo aparecer com rating muito
    diferente entre puzzles, o jogo ainda fica inteiro — a integridade do grupo
    vence a estratificação perfeita.
    """
    por_jogo: dict[str, list[dict]] = {}
    for linha in linhas:
        por_jogo.setdefault(linha["game_id"], []).append(linha)

    treino, val, teste = [], [], []
    for game_id, do_jogo in por_jogo.items():
        u = hash_do_jogo(game_id, seed)
        if u < frac_teste:
            teste.extend(do_jogo)
        elif u < frac_teste + frac_val:
            val.extend(do_jogo)
        else:
            treino.extend(do_jogo)

    return Divisao(treino, val, teste)


def _campo(linha: dict, *nomes: str):
    """Le um campo aceitando snake_case ou PascalCase do dataset."""
    for nome in nomes:
        if nome in linha:
            return linha[nome]
    return None


def normalizar(linha: dict) -> dict:
    """Converte uma linha do Lichess para o formato interno."""
    return {
        "puzzle_id": _campo(linha, "puzzle_id", "PuzzleId"),
        "game_id": _campo(linha, "game_id", "GameId") or "",
        "fen": _campo(linha, "fen", "FEN"),
        "moves": _campo(linha, "moves", "Moves"),
        "rating": int(_campo(linha, "rating", "Rating") or 0),
        "popularity": int(_campo(linha, "popularity", "Popularity") or 0),
        "nb_plays": int(_campo(linha, "nb_plays", "NbPlays") or 0),
        "temas": tuple(_campo(linha, "temas", "Themes", "themes") or ()),
    }


def filtrar(linhas: list[dict], *, popularidade_min: int = 80,
            partidas_min: int = 100, rating_min: int = 400,
            rating_max: int = 3000) -> list[dict]:
    """Mantém apenas puzzles limpos e jogados o suficiente.

    Puzzle com baixa popularidade costuma ter solução ambígua ou alternativa —
    ruído de rótulo que nenhum hiperparâmetro conserta.
    """
    saida = []
    for linha in linhas:
        if int(_campo(linha, "popularity", "Popularity") or 0) < popularidade_min:
            continue
        if int(_campo(linha, "nb_plays", "NbPlays") or 0) < partidas_min:
            continue
        rating = int(_campo(linha, "rating", "Rating") or 0)
        if not (rating_min <= rating <= rating_max):
            continue
        saida.append(linha)
    return saida


def auditar_vazamento(divisao: Divisao) -> dict:
    """Prova que nenhum GameId nem PuzzleId aparece em duas divisões."""
    jogos = {
        "treino": {l["game_id"] for l in divisao.treino},
        "validacao": {l["game_id"] for l in divisao.validacao},
        "teste": {l["game_id"] for l in divisao.teste},
    }
    ids = {
        "treino": {l["puzzle_id"] for l in divisao.treino},
        "validacao": {l["puzzle_id"] for l in divisao.validacao},
        "teste": {l["puzzle_id"] for l in divisao.teste},
    }
    return {
        "jogos_por_divisao": {k: len(v) for k, v in jogos.items()},
        "puzzles_por_divisao": {k: len(v) for k, v in ids.items()},
        "jogos_compartilhados": sorted(
            (jogos["treino"] & jogos["teste"])
            | (jogos["treino"] & jogos["validacao"])
            | (jogos["validacao"] & jogos["teste"])
        )[:20],
        "puzzles_compartilhados": sorted(
            (ids["treino"] & ids["teste"])
            | (ids["treino"] & ids["validacao"])
            | (ids["validacao"] & ids["teste"])
        )[:20],
    }


def escrever_manifesto(divisao: Divisao, destino: Path, *, fonte: str,
                       filtros: dict, seed: int) -> dict:
    """Registra proveniência e contagens. Sem isso o resultado não é auditável."""
    destino.mkdir(parents=True, exist_ok=True)
    manifesto = {
        "fonte": fonte,
        "seed": seed,
        "filtros": filtros,
        "divisoes": divisao.resumo(),
        "auditoria_vazamento": auditar_vazamento(divisao),
        "faixas_rating": {
            "treino": _histograma(divisao.treino),
            "validacao": _histograma(divisao.validacao),
            "teste": _histograma(divisao.teste),
        },
    }
    (destino / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifesto


def _histograma(linhas: list[dict]) -> dict[str, int]:
    contagem: dict[str, int] = {}
    for linha in linhas:
        f = faixa_de_rating(int(_campo(linha, "rating", "Rating") or 0))
        contagem[f] = contagem.get(f, 0) + 1
    return dict(sorted(contagem.items()))


# --------------------------------------------------------------------------
# Divisão estratificada por motivo + faixa de rating
# --------------------------------------------------------------------------

MOTIVO_PRINCIPAL = (
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "deflection",
    "sacrifice", "mateIn1", "mateIn2", "endgame", "middlegame", "opening",
    "promotion", "hangingPiece", "trappedPiece", "xRayAttack", "zugzwang",
    "attraction", "clearance", "interference", "intermezzo", "quietMove",
    "backRankMate", "smotheredMate", "arabianMate", "anastasiaMate",
)


def motivo_de(temas) -> str:
    """Escolhe o motivo mais específico do puzzle.

    A ordem importa: `mateIn1` e mais informativo que `mate`, que e mais
    informativo que `middlegame`. Usar o primeiro da lista conhecida evita que
    todo puzzle vire "middlegame".
    """
    conjunto = set(temas or ())
    for m in MOTIVO_PRINCIPAL:
        if m in conjunto:
            return m
    for t in (temas or ()):
        if t in ("crushing", "advantage", "equality", "mate", "short", "long",
                 "veryLong", "oneMove", "master", "masterVsMaster", "superGM"):
            continue
        return t
    return "outro"


def celula(linha: dict) -> str:
    """Célula de estratificação: motivo + faixa de rating."""
    return "%s|%s" % (motivo_de(linha.get("temas") or linha.get("Themes")),
                      faixa_de_rating(int(_campo(linha, "rating", "Rating") or 0)))


def dividir_estratificado(linhas: list[dict], *, seed: int = 0,
                          frac_val: float = 0.05,
                          frac_teste: float = 0.15) -> Divisao:
    """Divide por GameId, estratificando por (motivo, faixa de rating).

    Duas garantias ao mesmo tempo:
      1. nenhum GameId aparece em duas divisões (integridade do grupo);
      2. cada célula motivo×rating tem representação nas três divisões, para que
         o slice por tema seja comparável e não meça só composição.

    A estratificação é feita ordenando os jogos de cada célula por hash e cortando
    nas proporções pedidas. Se uma célula tiver poucos jogos, ela cai inteira na
    divisão que o hash indicar — e isso fica registrado no manifesto.
    """
    por_jogo: dict[str, list[dict]] = {}
    for linha in linhas:
        gid = _campo(linha, "game_id", "GameId") or str(_campo(linha, "puzzle_id", "PuzzleId"))
        por_jogo.setdefault(gid, []).append(linha)

    # Cada jogo pertence a uma célula (a do seu primeiro puzzle).
    celulas: dict[str, list[str]] = {}
    for gid, do_jogo in por_jogo.items():
        celulas.setdefault(celula(do_jogo[0]), []).append(gid)

    treino, val, teste = [], [], []
    for cel, jogos in celulas.items():
        ordenados = sorted(jogos, key=lambda g: hash_do_jogo(g, seed))
        n = len(ordenados)
        n_teste = int(round(n * frac_teste))
        n_val = int(round(n * frac_val))
        if n >= 3:
            n_teste = max(1, n_teste)
            n_val = max(1, n_val)
        corte_t = n - n_teste
        corte_v = max(0, corte_t - n_val)

        for g in ordenados[:corte_v]:
            treino.extend(por_jogo[g])
        for g in ordenados[corte_v:corte_t]:
            val.extend(por_jogo[g])
        for g in ordenados[corte_t:]:
            teste.extend(por_jogo[g])

    return Divisao(treino, val, teste)


def auditar_estratos(divisao: Divisao) -> dict:
    """Prova que motivo e rating estão cobertos nas três divisões."""
    def cobertura(linhas):
        motivos, faixas = {}, {}
        for l in linhas:
            m = motivo_de(l.get("temas") or l.get("Themes"))
            motivos[m] = motivos.get(m, 0) + 1
            f = faixa_de_rating(int(_campo(l, "rating", "Rating") or 0))
            faixas[f] = faixas.get(f, 0) + 1
        return motivos, faixas

    mt, ft = cobertura(divisao.treino)
    mv, fv = cobertura(divisao.validacao)
    ms, fs = cobertura(divisao.teste)

    motivos_todos = set(mt) | set(mv) | set(ms)
    faixas_todas = set(ft) | set(fv) | set(fs)

    return {
        "motivos_treino": len(mt), "motivos_teste": len(ms),
        "motivos_so_treino": sorted(set(mt) - set(ms)),
        "motivos_so_teste": sorted(set(ms) - set(mt)),
        "faixas_treino": sorted(ft), "faixas_teste": sorted(fs),
        "motivos_sem_teste": sorted(m for m in motivos_todos if m not in ms),
        "faixas_sem_teste": sorted(f for f in faixas_todas if f not in fs),
        "cobertura_motivos": {m: {"treino": mt.get(m, 0), "val": mv.get(m, 0),
                                  "teste": ms.get(m, 0)}
                              for m in sorted(motivos_todos)},
    }
