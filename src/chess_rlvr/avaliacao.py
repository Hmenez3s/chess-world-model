"""Métricas de avaliação com denominador explícito e incerteza.

Regra do projeto: nenhum número sem denominador. Toda taxa sai com intervalo de
confiança de Wilson, porque `3/4` e `300/400` não são a mesma evidência.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def wilson(sucessos: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de Wilson para uma proporção.

    Preferido ao intervalo normal porque não produz limites fora de [0,1] nem
    colapsa quando a proporção é 0 ou 1 — exatamente o caso do baseline.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = sucessos / total
    denom = 1 + z * z / total
    centro = (p + z * z / (2 * total)) / denom
    margem = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centro - margem), min(1.0, centro + margem))


@dataclass
class Contagem:
    """Contagem com denominador e intervalo."""
    sucessos: int = 0
    total: int = 0

    def somar(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.sucessos += 1

    @property
    def taxa(self) -> float:
        return self.sucessos / self.total if self.total else 0.0

    def linha(self, rotulo: str) -> str:
        lo, hi = wilson(self.sucessos, self.total)
        return "%-26s %5d/%-5d = %6.2f%%  IC95 [%.2f%%, %.2f%%]" % (
            rotulo, self.sucessos, self.total, 100 * self.taxa, 100 * lo, 100 * hi
        )


@dataclass
class ResultadoAvaliacao:
    """Agrega as métricas de uma execução de avaliação."""
    legal: Contagem = field(default_factory=Contagem)
    correto: Contagem = field(default_factory=Contagem)
    nao_parseavel: Contagem = field(default_factory=Contagem)
    por_faixa: dict[str, Contagem] = field(default_factory=dict)
    por_tema: dict[str, Contagem] = field(default_factory=dict)

    def registrar(self, *, status: str, faixa: str, temas: tuple[str, ...]) -> None:
        self.legal.somar(status in ("correto", "legal_bom", "legal_ruim"))
        self.correto.somar(status == "correto")
        self.nao_parseavel.somar(status == "nao_parseavel")

        self.por_faixa.setdefault(faixa, Contagem()).somar(status == "correto")
        for tema in temas:
            self.por_tema.setdefault(tema, Contagem()).somar(status == "correto")

    def relatorio(self, titulo: str) -> str:
        partes = [f"=== {titulo}", self.legal.linha("lance legal"),
                  self.correto.linha("lance correto"),
                  self.nao_parseavel.linha("nao parseavel")]

        if self.por_faixa:
            partes.append("-- por faixa de rating (correto) --")
            for faixa in sorted(self.por_faixa):
                partes.append(self.por_faixa[faixa].linha(faixa))

        # Só temas com massa suficiente: slice de 3 casos não é evidência.
        temas = [(t, c) for t, c in self.por_tema.items() if c.total >= 20]
        if temas:
            partes.append("-- por tema (correto, n>=20) --")
            for tema, cont in sorted(temas, key=lambda kv: -kv[1].taxa):
                partes.append(cont.linha(tema))

        return "\n".join(partes)

    def como_dict(self) -> dict:
        def bloco(c: Contagem) -> dict:
            lo, hi = wilson(c.sucessos, c.total)
            return {"sucessos": c.sucessos, "total": c.total,
                    "taxa": round(c.taxa, 4),
                    "ic95": [round(lo, 4), round(hi, 4)]}

        return {
            "legal": bloco(self.legal),
            "correto": bloco(self.correto),
            "nao_parseavel": bloco(self.nao_parseavel),
            "por_faixa": {k: bloco(v) for k, v in sorted(self.por_faixa.items())},
            "por_tema": {k: bloco(v) for k, v in sorted(self.por_tema.items())
                         if v.total >= 20},
        }


def correlacao_raciocinio_acerto(tem_raciocinio: list[bool], acertou: list[bool]) -> dict:
    """Mede se o raciocínio produzido tem relação com o acerto.

    Se a correlação for ~0, o texto gerado é decoração: o modelo acerta e erra
    independentemente do que "explica". Esse é um resultado publicável.
    """
    n = min(len(tem_raciocinio), len(acertou))
    if n == 0:
        return {"n": 0}
    x = [1.0 if v else 0.0 for v in tem_raciocinio[:n]]
    y = [1.0 if v else 0.0 for v in acertou[:n]]
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / n)
    r = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0
    return {"n": n, "correlacao": round(r, 4),
            "taxa_acerto_com_raciocinio": round(sum(b for a, b in zip(x, y) if a) / max(1, sum(x)), 4),
            "taxa_acerto_sem_raciocinio": round(sum(b for a, b in zip(x, y) if not a) / max(1, n - sum(x)), 4)}
