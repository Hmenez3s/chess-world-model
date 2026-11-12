#!/usr/bin/env python3
"""Mede se o raciocínio produzido tem relação com o acerto.

Pergunta de pesquisa: quando o modelo "pensa antes de responder", ele acerta mais?
Se a correlação for ~0, o texto é racionalização post-hoc — decoração que parece
raciocínio. Esse é um resultado publicável, inclusive negativo.

Duas condições:
  A) resposta direta (só o lance)      -> referência de acerto
  B) resposta com raciocínio explícito -> mede se pensar ajuda

A recompensa do treino nunca olhou o texto, só o lance. Por isso a comparação é
legítima: o modelo não foi otimizado a produzir texto convincente.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr.avaliacao import ResultadoAvaliacao, correlacao_raciocinio_acerto
from chess_rlvr.dados import faixa_de_rating, normalizar
from chess_rlvr.verificador import (
    PuzzleInvalido, avaliar_lance, montar_puzzle, normalizar_uci,
)

PROMPT_DIRETO = (
    "Você joga com as {lado} e é a sua vez.\n\n"
    "Posição (peças por casa):\n{posicao}\n\n"
    "FEN: {fen}\n\n"
    "Responda com o melhor lance em notação UCI (ex.: e2e4). "
    "Escreva apenas o lance."
)

PROMPT_COT = (
    "Você joga com as {lado} e é a sua vez.\n\n"
    "Posição (peças por casa):\n{posicao}\n\n"
    "FEN: {fen}\n\n"
    "Raciocine assim:\n"
    "Situação: material e xeque\n"
    "Candidatos: liste 2 ou 3 lances com o que cada um faz\n"
    "Comparação: por que o escolhido é o melhor\n"
    "LANCE: <uci>\n\n"
    "Use apenas casas de a1 a h8."
)


FORMATO = "candidatos"   # ajustado por --formato no main


def montar(tabuleiro, com_raciocinio: bool) -> str:
    from chess_rlvr.verificador import descrever_posicao
    import chess
    lado = "brancas" if tabuleiro.turn == chess.WHITE else "pretas"
    if com_raciocinio and FORMATO == "forcados":
        from chess_rlvr.raciocinio import prompt_forcados
        return prompt_forcados(tabuleiro)
    tpl = PROMPT_COT if com_raciocinio else PROMPT_DIRETO
    return tpl.format(lado=lado, posicao=descrever_posicao(tabuleiro), fen=tabuleiro.fen())


def extrair(texto: str, com_raciocinio: bool) -> tuple[str | None, bool, int]:
    """Devolve (lance, tem_raciocinio, tokens_de_raciocinio)."""
    if not com_raciocinio:
        return normalizar_uci(texto), False, 0

    m = re.search(r"LANCE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", texto, re.IGNORECASE)
    if m:
        lance = m.group(1).lower()
    else:
        lance = normalizar_uci(texto)

    # Raciocínio = o texto que não é o lance final.
    sem_lance = re.sub(r"LANCE:\s*[a-h][1-8][a-h][1-8][qrbn]?", "", texto, flags=re.IGNORECASE)
    palavras = len(sem_lance.split())
    return lance, palavras >= 15, palavras


_PADRAO_INVALIDO = re.compile(r"\b[i-r][1-9]\b|\b[a-h]0\b|\b[a-h](?:1[0-9]|[2-9][0-9])\b")


def sanidade_do_raciocinio(texto: str) -> bool:
    """Detecta notação impossível — colunas i-r ou fileiras fora de 1-8.

    Um raciocínio que cita casas inexistentes não é raciocínio: é alucinação.
    Medir correlação sem checar isso compara texto válido com lixo.
    """
    return _PADRAO_INVALIDO.search(texto) is None


def carregar(quantidade: int, seed: int):
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    saida = []
    for bruto in ds:
        if len(saida) >= quantidade:
            break
        l = normalizar(bruto)
        try:
            saida.append(montar_puzzle(
                puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                rating=l["rating"], temas=l["temas"], game_id=l["game_id"]))
        except PuzzleInvalido:
            continue
    return saida


def gerar(modelo, tok, prompts, device, max_novos, lote):
    out = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), lote):
        bloco = prompts[i:i + lote]
        msgs = [[{"role": "user", "content": p}] for p in bloco]
        ent = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", padding=True,
                                      return_dict=True).to(device)
        with torch.no_grad():
            r = modelo.generate(**ent, max_new_tokens=max_novos, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(r[:, ent["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--formato", choices=["candidatos", "forcados"],
                    default="candidatos")
    ap.add_argument("--saida", default="out/cot.json")
    args = ap.parse_args()
    global FORMATO
    FORMATO = args.formato

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, dtype=torch.bfloat16, device_map=device)
    if args.adapter:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, args.adapter)
    modelo.eval()

    puzzles = carregar(args.puzzles, args.seed)
    print("puzzles:", len(puzzles), flush=True)

    resultados = {}
    for rotulo, com_cot, max_novos in (("direto", False, 16), ("raciocinio", True, 320)):
        prompts = [montar(p.tabuleiro, com_cot) for p in puzzles]
        respostas = gerar(modelo, tok, prompts, device, max_novos, args.lote)

        res = ResultadoAvaliacao()
        tem_cot, acertou, tokens_cot = [], [], []
        sanidade = {"validos": 0, "invalidos": 0}
        for p, texto in zip(puzzles, respostas):
            if com_cot:
                if sanidade_do_raciocinio(texto):
                    sanidade["validos"] += 1
                else:
                    sanidade["invalidos"] += 1
            lance, tem, n_pal = extrair(texto, com_cot)
            r = avaliar_lance(tabuleiro=p.tabuleiro, lance_previsto=lance,
                              lance_solucao=p.primeiro_lance)
            res.registrar(status=r.status, faixa=faixa_de_rating(p.rating), temas=p.temas)
            tem_cot.append(tem)
            acertou.append(r.status == "correto")
            tokens_cot.append(n_pal)

        print()
        print(res.relatorio("CONDICAO: " + rotulo))
        resultados[rotulo] = {
            "resumo": res.como_dict(),
            "amostra": respostas[:5],
            "sanidade": sanidade if com_cot else None,
        }
        if com_cot:
            corr = correlacao_raciocinio_acerto(tem_cot, acertou)
            corr["media_palavras"] = round(sum(tokens_cot) / max(1, len(tokens_cot)), 1)
            print()
            print("=== RACIOCINIO x ACERTO ===")
            print("n                        :", corr["n"])
            print("correlacao               : %+.4f" % corr["correlacao"])
            print("acerto COM raciocinio    : %.2f%%" % (100 * corr["taxa_acerto_com_raciocinio"]))
            print("acerto SEM raciocinio    : %.2f%%" % (100 * corr["taxa_acerto_sem_raciocinio"]))
            print("media de palavras        :", corr["media_palavras"])
            print("raciocinio com notacao valida: %d/%d = %.1f%%" % (
                sanidade["validos"], sanidade["validos"] + sanidade["invalidos"],
                100 * sanidade["validos"] / max(1, sanidade["validos"] + sanidade["invalidos"])))
            resultados["correlacao"] = corr

    d = resultados["direto"]["resumo"]["correto"]
    c = resultados["raciocinio"]["resumo"]["correto"]
    print()
    print("=== COMPARACAO ===")
    print("direto      : %d/%d = %.2f%%" % (d["sucessos"], d["total"], 100 * d["taxa"]))
    print("raciocinio  : %d/%d = %.2f%%" % (c["sucessos"], c["total"], 100 * c["taxa"]))

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
