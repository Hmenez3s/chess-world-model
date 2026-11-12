#!/usr/bin/env python3
"""Estima Elo de puzzle por máxima verossimilhança e bootstrap de itens."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def probabilidade(rating: int, elo: float) -> float:
    expoente = max(-700.0, min(700.0, math.log(10) * (rating - elo) / 400))
    return 1 / (1 + math.exp(expoente))


def estimar(itens: list[dict]) -> float | None:
    """Raiz da derivada da log-verossimilhança; None para amostra extrema/vazia."""
    n = len(itens)
    acertos = sum(bool(item["correto"]) for item in itens)
    if n == 0 or acertos in (0, n):
        return None

    def derivada(elo: float) -> float:
        return sum((int(bool(item["correto"])) - probabilidade(int(item["rating"]), elo))
                   for item in itens)

    baixo, alto = -1000.0, 4000.0
    for _ in range(100):
        meio = (baixo + alto) / 2
        if derivada(meio) > 0:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    pos = (len(ordenados) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordenados[lo] + (ordenados[hi] - ordenados[lo]) * (pos - lo)


def limite_superior_zero(itens: list[dict], z: float = 1.96) -> float | None:
    """Com 0 acertos o MLE vai a -infinito; o dado ainda limita o Elo por cima.

    Limite superior de Wilson para 0/n: z^2 / (n + z^2). Procura o R em que a
    taxa de acerto esperada sobre os ratings reais do teste iguala esse limite.
    Um modelo acima desse R teria acertado algum puzzle com 95% de chance.
    """
    n = len(itens)
    if n == 0:
        return None
    alvo = z * z / (n + z * z)

    def esperado(elo: float) -> float:
        return sum(probabilidade(int(item["rating"]), elo) for item in itens) / n

    baixo, alto = -2000.0, 4000.0
    for _ in range(100):
        meio = (baixo + alto) / 2
        if esperado(meio) < alvo:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def analisar(caminho: str, bootstrap: int, rng: random.Random) -> dict:
    with open(caminho, encoding="utf-8") as fh:
        dados = json.load(fh)
    itens = dados["itens"]
    for item in itens:
        if "rating" not in item or "correto" not in item:
            raise ValueError(f"{caminho}: cada item precisa de rating e correto")
    n = len(itens)
    acertos = sum(bool(item["correto"]) for item in itens)
    taxa = acertos / n if n else None
    elo = estimar(itens)
    if elo is None:
        razao = ("sem itens" if n == 0 else
                 "0 acertos: máxima verossimilhança tende a -infinito" if acertos == 0 else
                 "100% de acertos: máxima verossimilhança tende a +infinito")
        ic = None
    else:
        amostras = []
        for _ in range(bootstrap):
            amostra = [itens[rng.randrange(n)] for _ in range(n)]
            valor = estimar(amostra)
            if valor is not None:
                amostras.append(valor)
        ic = ([percentil(amostras, 0.025), percentil(amostras, 0.975)]
              if amostras else None)
        razao = None
    limite = limite_superior_zero(itens) if (n and acertos == 0) else None
    return {"arquivo": caminho, "n": n, "acertos": acertos, "taxa": taxa,
            "elo_puzzle": elo, "ic95": ic, "elo_maximo": limite,
            "status": "estimado" if elo is not None else "indefinido",
            "razao": razao, "bootstrap_validos": (len(amostras) if elo is not None else 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arquivos", nargs="+")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default="out/elo-puzzle.json")
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap deve ser >= 0")
    rng = random.Random(args.seed)
    resultados = [analisar(path, args.bootstrap, rng) for path in args.arquivos]
    print(f"{'arquivo':<32} {'n':>6} {'acertos':>8} {'taxa':>9} {'Elo puzzle':>12} IC95")
    for r in resultados:
        taxa = "indef." if r["taxa"] is None else f"{100*r['taxa']:.2f}%"
        elo = ("indefinido" if r["elo_puzzle"] is None else f"{r['elo_puzzle']:.1f}")
        if r["elo_puzzle"] is None and r.get("elo_maximo") is not None:
            elo = f"<= {r['elo_maximo']:.0f}"
        ic = "indefinido" if r["ic95"] is None else f"[{r['ic95'][0]:.1f}, {r['ic95'][1]:.1f}]"
        print(f"{r['arquivo']:<32} {r['n']:>6} {r['acertos']:>8} {taxa:>9} {elo:>12} {ic}")
        if r["razao"]:
            print("  motivo:", r["razao"])
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
