#!/usr/bin/env python3
"""Gera as curvas de treino a partir dos logs reais das execuções.

Nada aqui é inventado: cada ponto vem de `out/*-metricas.json`, gravado pelo trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/curvas")
OUT.mkdir(parents=True, exist_ok=True)


def serie(log, chave):
    xs, ys = [], []
    for e in log:
        if isinstance(e, dict) and chave in e and "epoch" in e:
            v = e[chave]
            try:
                ys.append(float(v))
                xs.append(float(e["epoch"]))
            except (TypeError, ValueError):
                continue
    return xs, ys


def plotar_raciocinio():
    """Curva do GRPO sobre o traço de raciocínio, do log real.

    Mostra as três coisas que decidem se o RLVR está funcionando:
    recompensa subindo, variância de grupo (frac_reward_zero_std) e
    quanto do orçamento de tokens é perdido em truncamento.
    """
    caminho = Path("out/grpo-cot-metricas.json")
    if not caminho.exists():
        print("grpo-cot-metricas.json ausente, pulando")
        return
    d = json.loads(caminho.read_text())
    log = [e for e in d.get("log", []) if isinstance(e, dict) and "step" in e]
    if not log:
        print("log vazio, pulando raciocinio")
        return
    passos = [e["step"] for e in log]

    def num(e, k, alt=None):
        v = e.get(k, alt)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    recompensa = [num(e, "reward") for e in log]
    zero_std = [num(e, "frac_reward_zero_std") for e in log]
    clip = [num(e, "completions/clipped_ratio") for e in log]

    fig, eixos = plt.subplots(1, 3, figsize=(14, 4))
    if any(v is not None for v in recompensa):
        eixos[0].plot(passos, [v if v is not None else float("nan") for v in recompensa],
                      marker="o", ms=3, color="#2563eb")
        eixos[0].set_title("recompensa total")
        eixos[0].set_xlabel("passo")
        eixos[0].grid(alpha=0.3)

    if any(v is not None for v in zero_std):
        eixos[1].plot(passos, [v if v is not None else float("nan") for v in zero_std],
                      marker="o", ms=3, color="#16a34a")
        eixos[1].axhline(0, ls="--", lw=1, color="gray")
        eixos[1].set_title("grupos com variância zero (0 = todo grupo tem sinal)")
        eixos[1].set_xlabel("passo")
        eixos[1].grid(alpha=0.3)

    if any(v is not None for v in clip):
        eixos[2].plot(passos, [v if v is not None else float("nan") for v in clip],
                      marker="o", ms=3, color="#dc2626")
        eixos[2].set_title("gerações truncadas em %d tokens" % d.get("max_completude", 0))
        eixos[2].set_xlabel("passo")
        eixos[2].set_ylim(0, 1.02)
        eixos[2].grid(alpha=0.3)

    fig.suptitle("GRPO sobre raciocínio (RLVR) — VRAM pico %.2f GB"
                 % d.get("vram_pico_gb", 0))
    fig.tight_layout()
    fig.savefig(OUT / "raciocinio.png", dpi=130)
    print("raciocinio.png")


def plotar_raciocinio_sft():
    """Loss do SFT de raciocínio sobre o adapter de movimentos."""
    caminho = Path("out/sft-cot2-metricas.json")
    if not caminho.exists():
        print("sft-cot2-metricas.json ausente, pulando")
        return
    d = json.loads(caminho.read_text())
    log = [e for e in d.get("log", []) if isinstance(e, dict) and "loss" in e]
    if not log:
        return
    xs = list(range(len(log)))
    perda = [float(e["loss"]) for e in log]
    acerto_tok = [float(e.get("mean_token_accuracy", 0)) for e in log]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(xs, perda, marker="o", ms=3, color="#2563eb", label="loss")
    ax1.set_xlabel("registro")
    ax1.set_ylabel("loss", color="#2563eb")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, acerto_tok, marker="s", ms=3, color="#16a34a",
             label="acurácia por token")
    ax2.set_ylabel("acurácia por token", color="#16a34a")
    ax1.set_title("SFT de raciocínio sobre o adapter (%d traços, VRAM %.2f GB)"
                  % (d.get("tracos", 0), d.get("vram_pico_gb", 0)))
    fig.tight_layout()
    fig.savefig(OUT / "sft-raciocinio.png", dpi=130)
    print("sft-raciocinio.png")


def plotar_grpo():
    d = json.loads(Path("out/grpo-metricas.json").read_text())
    log = d.get("log", [])
    if not log:
        print("sem log de GRPO")
        return

    fig, eixos = plt.subplots(1, 3, figsize=(15, 4))

    for chave, rotulo, cor in (
        ("rewards/recompensa_correcao/mean", "acerto (correção)", "#2E7D32"),
        ("rewards/recompensa_legalidade/mean", "legalidade", "#1565C0"),
        ("entropy", "entropia", "#C62828"),
    ):
        xs, ys = serie(log, chave)
        if xs:
            eixos[0].plot(xs, ys, marker="o", ms=3, label=rotulo, color=cor)
    eixos[0].set_xlabel("época")
    eixos[0].set_title("Recompensa e entropia")
    eixos[0].legend(fontsize=8)
    eixos[0].grid(alpha=0.3)

    xs, ys = serie(log, "frac_reward_zero_std")
    if xs:
        eixos[1].plot(xs, ys, marker="o", ms=3, color="#6A1B9A")
        eixos[1].set_ylabel("fração de grupos com variância zero")
        eixos[1].set_title("Sinal de aprendizado disponível")
        eixos[1].set_ylim(0, 1)
        eixos[1].grid(alpha=0.3)

    xs, ys = serie(log, "kl")
    if xs:
        eixos[2].plot(xs, ys, marker="o", ms=3, color="#EF6C00")
        eixos[2].set_ylabel("KL")
        eixos[2].set_title("Divergência da política de referência")
        eixos[2].grid(alpha=0.3)

    for ax in eixos:
        ax.set_xlabel("época")
    fig.suptitle("GRPO — Qwen2.5-1.5B em puzzles de xadrez", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "grpo.png", dpi=130)
    print("grpo.png")


def plotar_sft():
    d = json.loads(Path("out/sft-metricas.json").read_text())
    log = d.get("estado_log", [])
    if not log:
        print("sem log de SFT")
        return
    xs, perda = serie(log, "loss")
    _, acerto = serie(log, "mean_token_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, perda, marker="o", color="#1565C0", label="loss")
    ax.set_xlabel("passo")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    if acerto:
        ax2.plot(xs[:len(acerto)], acerto, marker="s", color="#2E7D32",
                 label="acurácia de token")
        ax2.set_ylabel("acurácia de token")
    ax.set_title("SFT de warmup — loss e acurácia")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sft.png", dpi=130)
    print("sft.png")


def plotar_comparativo():
    """Barras com IC de Wilson: base vs SFT vs GRPO, por faixa de rating."""
    arqs = {
        "SFT": "out/aval-sft600.json",
        "GRPO": "out/aval-grpo600.json",
    }
    dados = {}
    for nome, caminho in arqs.items():
        p = Path(caminho)
        if p.exists():
            dados[nome] = json.loads(p.read_text())["original"]["por_faixa"]

    if not dados:
        print("sem avaliações comparáveis")
        return

    faixas = sorted(set().union(*[set(d) for d in dados.values()]),
                    key=lambda f: int(f.split("-")[0].rstrip("+")))
    faixas = [f for f in faixas if any(d.get(f, {}).get("total", 0) >= 20 for d in dados.values())]

    largura = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (nome, d) in enumerate(dados.items()):
        xs, ys, err = [], [], []
        for j, f in enumerate(faixas):
            cel = d.get(f, {})
            if cel.get("total", 0) < 20:
                continue
            xs.append(j + (i - 0.5) * largura)
            ys.append(100 * cel["taxa"])
            lo, hi = cel["ic95"]
            err.append([100 * (cel["taxa"] - lo), 100 * (hi - cel["taxa"])])
        ax.bar(xs, ys, largura, label=nome,
               yerr=list(zip(*err)) if err else None, capsize=3)

    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels(faixas, rotation=20)
    ax.set_ylabel("acerto (%)")
    ax.set_xlabel("faixa de rating")
    ax.set_title("Acerto por dificuldade — SFT vs GRPO (n=600, IC95)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "comparativo.png", dpi=130)
    print("comparativo.png")


if __name__ == "__main__":
    plotar_sft()
    plotar_grpo()
    plotar_comparativo()
    plotar_raciocinio()
    plotar_raciocinio_sft()
