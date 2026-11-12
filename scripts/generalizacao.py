#!/usr/bin/env python3
"""Compara generalização em Lig 4 e GSM8K usando o modelo base ou um adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.lig4 import colunas_legais, desenhar, gerar_posicoes


def wilson(sucessos: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = sucessos / total
    d = 1 + z * z / total
    centro = (p + z * z / (2 * total)) / d
    margem = (z / d) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return 100 * max(0.0, centro - margem), 100 * min(1.0, centro + margem)


def prompt_lig4(posicao: dict) -> str:
    return (
        "Você está jogando Lig 4. As peças caem até a casa livre mais baixa da coluna. "
        "Vence quem alinhar 4 peças na horizontal, vertical ou diagonal.\n\n"
        f"{desenhar(posicao['tabuleiro'])}\n\n"
        f"É a vez de {posicao['lado']}.\n"
        "Responda apenas com o número da coluna (1 a 7)."
    )


def extrair_coluna(texto: str) -> int | None:
    match = re.search(r"[1-7]", texto)
    return int(match.group()) if match else None


def _numero(texto: str) -> str | None:
    match = re.search(r"-?\d[\d,]*", texto)
    return match.group().replace(",", "") if match else None


def extrair_resposta_gsm(texto: str) -> str | None:
    match = re.search(r"Resposta\s*:\s*([^\n]+)", texto, flags=re.IGNORECASE)
    if match:
        numero = _numero(match.group(1))
        if numero is not None:
            return numero
    encontrados = re.findall(r"-?\d[\d,]*", texto)
    return encontrados[-1].replace(",", "") if encontrados else None


def _carregar_modelo(nome: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nome)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = AutoModelForCausalLM.from_pretrained(
        nome, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, adapter)
    modelo.eval()
    return modelo, tok


def _gerar(modelo, tok, prompts: list[str], max_novos: int, lote: int) -> list[str]:
    import torch

    respostas = []
    for inicio in range(0, len(prompts), lote):
        bloco = prompts[inicio:inicio + lote]
        mensagens = [[{"role": "user", "content": prompt}] for prompt in bloco]
        textos = [tok.apply_chat_template(
            mensagem, tokenize=False, add_generation_prompt=True) for mensagem in mensagens]
        entradas = tok(textos, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            saida = modelo.generate(**entradas, max_new_tokens=max_novos,
                                    do_sample=False, pad_token_id=tok.eos_token_id)
        tamanho_prompt = entradas.input_ids.shape[1]
        respostas.extend(tok.decode(saida[i][tamanho_prompt:], skip_special_tokens=True)
                         for i in range(len(bloco)))
        print("  %d/%d" % (len(respostas), len(prompts)), flush=True)
    return respostas


def _avaliar_lig4(modelo, tok, args) -> dict:
    total = args.lig4
    por_tipo = {"vencer": (total + 1) // 2, "bloquear": total // 2}
    resultado = {}
    todas = []
    for tipo, quantidade in por_tipo.items():
        posicoes = gerar_posicoes(quantidade, tipo, args.seed)
        textos = _gerar(modelo, tok, [prompt_lig4(p) for p in posicoes], 8, args.lote)
        acertos = sum(extrair_coluna(t) in p["corretas"] for p, t in zip(posicoes, textos))
        acaso = (sum(len(p["corretas"]) / len(colunas_legais(p["tabuleiro"]))
                     for p in posicoes) / len(posicoes)) if posicoes else 0.0
        resultado[tipo] = {
            "n": len(posicoes), "acertos": acertos,
            "acerto_pct": 100 * acertos / len(posicoes) if posicoes else 0.0,
            "ic95": wilson(acertos, len(posicoes)), "acaso_medio_pct": 100 * acaso,
            "amostras": [
                {"tabuleiro": desenhar(p["tabuleiro"]), "lado": p["lado"],
                 "corretas": sorted(p["corretas"]), "resposta": t}
                for p, t in zip(posicoes[:3], textos[:3])
            ],
        }
        todas.extend(zip(posicoes, textos))
    acertos_total = sum(extrair_coluna(t) in p["corretas"] for p, t in todas)
    n_total = len(todas)
    acaso_total = (sum(len(p["corretas"]) / len(colunas_legais(p["tabuleiro"]))
                       for p, _ in todas) / n_total) if n_total else 0.0
    resultado["total"] = {
        "n": n_total, "acertos": acertos_total,
        "acerto_pct": 100 * acertos_total / n_total if n_total else 0.0,
        "ic95": wilson(acertos_total, n_total), "acaso_medio_pct": 100 * acaso_total,
    }
    return resultado


def _avaliar_gsm8k(modelo, tok, quantidade: int, lote: int) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    itens = list(dataset.select(range(min(max(0, quantidade), len(dataset)))))
    prompts = [
        item["question"] + "\n\nResolva passo a passo e termine com a linha 'Resposta: <número>'."
        for item in itens
    ]
    textos = _gerar(modelo, tok, prompts, 320, lote)
    acertos = 0
    amostras = []
    for item, texto in zip(itens, textos):
        gabarito = item["answer"].split("####")[-1].strip()
        esperado = _numero(gabarito)
        previsto = extrair_resposta_gsm(texto)
        acertos += previsto is not None and previsto == esperado
        if len(amostras) < 3:
            amostras.append({"questao": item["question"], "esperado": esperado,
                             "resposta_extraida": previsto, "texto": texto[:1000]})
    n = len(itens)
    return {"n": n, "acertos": acertos,
            "acerto_pct": 100 * acertos / n if n else 0.0,
            "ic95": wilson(acertos, n), "amostras": amostras}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--lig4", type=int, default=200)
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/generalizacao.json")
    args = ap.parse_args()
    if args.lig4 < 0 or args.gsm8k < 0 or args.lote < 1:
        ap.error("--lig4 e --gsm8k devem ser >= 0, e --lote deve ser >= 1")

    modelo, tok = _carregar_modelo(args.modelo, args.adapter)
    print("Avaliando Lig 4...", flush=True)
    lig4 = _avaliar_lig4(modelo, tok, args)
    print("Avaliando GSM8K...", flush=True)
    gsm8k = _avaliar_gsm8k(modelo, tok, args.gsm8k, args.lote)
    resultado = {"modelo": args.modelo, "adapter": args.adapter, "seed": args.seed,
                 "lig4": lig4, "gsm8k": gsm8k}

    print("\n=== RESUMO DE GENERALIZAÇÃO ===")
    for tipo in ("vencer", "bloquear", "total"):
        r = lig4[tipo]
        print("Lig 4 %-8s: %d/%d = %.2f%% IC95 [%.2f; %.2f], acaso %.2f%%" % (
            tipo, r["acertos"], r["n"], r["acerto_pct"], *r["ic95"],
            r["acaso_medio_pct"]))
    print("GSM8K      : %d/%d = %.2f%% IC95 [%.2f; %.2f]" % (
        gsm8k["acertos"], gsm8k["n"], gsm8k["acerto_pct"], *gsm8k["ic95"]))
    caminho = Path(args.saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", caminho)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
