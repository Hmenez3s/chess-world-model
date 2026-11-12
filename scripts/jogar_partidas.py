#!/usr/bin/env python3
"""Joga partidas em lote contra um oponente de referência."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import chess
import chess.engine


def carregar_modelo(modelo_id: str, adapter: str):
    # Importação tardia mantém o jogador falso executável apenas com python-chess.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modelo_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_id, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        modelo = PeftModel.from_pretrained(modelo, adapter)
    modelo.eval()
    return tok, modelo, torch


def gerar_lances(jogos, tok, modelo, torch, formato: str,
                 amostrar: bool) -> list[tuple[str, str]]:
    from chess_rlvr import mav
    from chess_rlvr.verificador import montar_prompt, normalizar_uci

    prompts = []
    for jogo in jogos:
        conteudo = (mav.prompt_posicao(jogo["board"].fen()) if formato == "mav"
                    else montar_prompt(jogo["board"]))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": conteudo}], tokenize=False,
            add_generation_prompt=True))
    encoded = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    max_tokens = 440 if formato == "mav" else 16
    opcoes = {"max_new_tokens": max_tokens, "do_sample": amostrar,
              "pad_token_id": tok.eos_token_id}
    if amostrar:
        opcoes["temperature"] = 0.7
    with torch.no_grad():
        saidas = modelo.generate(**encoded, **opcoes)
    resultados = []
    for i, jogo in enumerate(jogos):
        texto = tok.decode(saidas[i][encoded.input_ids.shape[1]:], skip_special_tokens=True)
        uci = mav.extrair_campos(texto)["melhor"] if formato == "mav" else normalizar_uci(texto)
        resultados.append((uci or "", texto))
    return resultados


def gerar_em_lotes(jogos, tok, modelo, torch, formato: str, amostrar: bool,
                   tamanho: int) -> list[tuple[str, str]]:
    saidas = []
    for inicio in range(0, len(jogos), tamanho):
        saidas.extend(gerar_lances(jogos[inicio:inicio + tamanho], tok, modelo,
                                   torch, formato, amostrar))
    return saidas


def intervalo_wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margem = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centro - margem), min(1.0, centro + margem)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--formato", choices=["mav", "direto"], default="mav")
    ap.add_argument("--partidas", type=int, default=40)
    ap.add_argument("--oponente", choices=["aleatorio", "stockfish"], default="aleatorio")
    ap.add_argument("--elo", type=int, default=1320)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-lances", type=int, default=200)
    ap.add_argument("--saida", default="out/partidas.json")
    ap.add_argument("--verificar", action="store_true",
                    help="usa o proprio top5/legais do modelo para corrigir lance ilegal")
    ap.add_argument("--jogador-falso", choices=["aleatorio"])
    args = ap.parse_args()
    if args.partidas <= 0 or args.lote <= 0 or args.max_lances <= 0:
        ap.error("partidas, lote e max-lances devem ser positivos")

    rng = random.Random(0)
    tok = modelo = torch = None
    if not args.jogador_falso:
        raiz = os.path.join(os.path.dirname(__file__), "..", "src")
        import sys
        sys.path.insert(0, raiz)
        tok, modelo, torch = carregar_modelo(args.modelo, args.adapter)

    motor = None
    if args.oponente == "stockfish":
        caminho = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/bin/stockfish"))
        motor = chess.engine.SimpleEngine.popen_uci(caminho)
        motor.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})

    jogos = [{"board": chess.Board(), "modelo_brancas": i % 2 == 0,
              "uci": [], "terminado": False, "resultado": None, "motivo": None}
             for i in range(args.partidas)]
    lances_modelo = lances_ilegais = 0
    origens_verificacao: dict[str, int] = {}
    try:
        while any(not j["terminado"] for j in jogos):
            # Agrupa todas as posições em que o modelo joga nesta rodada.
            # Um tabuleiro terminado nao tem lances legais: sem esta guarda o
            # fallback aleatorio quebra com "Cannot choose from an empty sequence".
            pendentes = [j for j in jogos if not j["terminado"]
                         and not j["board"].is_game_over(claim_draw=True)
                         and j["board"].turn == (chess.WHITE if j["modelo_brancas"] else chess.BLACK)]
            escolhas = ([(rng.choice(list(j["board"].legal_moves)).uci(), "")
                         for j in pendentes]
                        if args.jogador_falso else
                        gerar_em_lotes(pendentes, tok, modelo, torch, args.formato,
                                       False, args.lote))
            def e_legal(jogo, uci):
                try:
                    return chess.Move.from_uci(uci) in jogo["board"].legal_moves
                except (ValueError, TypeError):
                    return False

            retry = [k for k, (j, (uci, _t)) in enumerate(zip(pendentes, escolhas))
                     if not e_legal(j, uci)]
            if retry and not args.jogador_falso:
                alternativas = gerar_em_lotes([pendentes[k] for k in retry], tok, modelo,
                                              torch, args.formato, True, args.lote)
                for k, par in zip(retry, alternativas):
                    escolhas[k] = par
            for j, (uci, texto) in zip(pendentes, escolhas):
                if j["board"].is_game_over(claim_draw=True):
                    continue
                lances_modelo += 1
                try:
                    lance = chess.Move.from_uci(uci)
                except (ValueError, TypeError):
                    lance = None
                if lance not in j["board"].legal_moves:
                    if args.verificar and args.formato == "mav" and texto:
                        from chess_rlvr.mav import lance_verificado
                        alternativo, origem = lance_verificado(texto, j["board"])
                        if alternativo:
                            lance = chess.Move.from_uci(alternativo)
                            origens_verificacao[origem] = origens_verificacao.get(origem, 0) + 1
                    if lance not in j["board"].legal_moves:
                        lances_ilegais += 1
                        lance = rng.choice(list(j["board"].legal_moves))
                j["board"].push(lance)
                j["uci"].append(lance.uci())

            # Os lances do oponente são aplicados depois dos lances do modelo.
            for j in jogos:
                if j["terminado"]:
                    continue
                if j["board"].is_game_over(claim_draw=True):
                    outcome = j["board"].outcome(claim_draw=True)
                    if outcome.winner is None:
                        j["resultado"], j["motivo"] = "E", "empate_regras"
                    else:
                        venceu_modelo = outcome.winner == (chess.WHITE if j["modelo_brancas"] else chess.BLACK)
                        j["resultado"] = "V" if venceu_modelo else "D"
                        j["motivo"] = "xeque-mate" if j["board"].is_checkmate() else "regras"
                    j["terminado"] = True
                elif j["board"].ply() >= args.max_lances:
                    j["resultado"], j["motivo"], j["terminado"] = "E", "adjudicacao_max_lances", True

            oponentes = [j for j in jogos if not j["terminado"] and
                         j["board"].turn != (chess.WHITE if j["modelo_brancas"] else chess.BLACK)]
            for j in oponentes:
                if args.oponente == "aleatorio":
                    lance = rng.choice(list(j["board"].legal_moves))
                else:
                    lance = motor.play(j["board"], chess.engine.Limit(time=0.05)).move
                j["board"].push(lance)
                j["uci"].append(lance.uci())
    finally:
        if motor is not None:
            motor.quit()

    v = sum(j["resultado"] == "V" for j in jogos)
    e = sum(j["resultado"] == "E" for j in jogos)
    d = sum(j["resultado"] == "D" for j in jogos)
    pontos = (v + 0.5 * e) / args.partidas
    resultado = {"partidas": [{"cor_modelo": "brancas" if j["modelo_brancas"] else "pretas",
                                "resultado": j["resultado"], "motivo": j["motivo"], "uci": j["uci"]}
                               for j in jogos],
                 "vitorias": v, "empates": e, "derrotas": d, "pontuacao": pontos,
                 "lances_modelo": lances_modelo, "lances_ilegais": lances_ilegais,
                 "taxa_lances_ilegais": lances_ilegais / lances_modelo if lances_modelo else 0.0,
                 "verificado": args.verificar,
                 "origens_verificacao": origens_verificacao}
    if args.oponente == "stockfish":
        if 0 < pontos < 1:
            lo, hi = intervalo_wilson(pontos, args.partidas)
            transformar = lambda score: args.elo + 400 * math.log10(score / (1 - score))
            resultado["elo_estimado"] = args.elo + 400 * math.log10(pontos / (1 - pontos))
            resultado["elo_ic95"] = [transformar(lo), transformar(hi)]
        else:
            # Pontuacao 0 ou 1 nao tem Elo (log divergiria). O limite superior
            # dado pelo IC de Wilson ainda informa: "Elo no maximo X".
            resultado["elo_estimado"] = None
            resultado["elo_ic95"] = None
            resultado["elo_status"] = "indefinido: pontuação 0 ou 1"
            lo, hi = intervalo_wilson(pontos, args.partidas)
            limite = lo if pontos == 1.0 else hi
            if 0 < limite < 1:
                resultado["elo_limite"] = args.elo + 400 * math.log10(limite / (1 - limite))
                resultado["elo_limite_lado"] = "máximo" if pontos == 0 else "mínimo"
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V/E/D: {v}/{e}/{d}; pontuação: {pontos:.3f}; "
          f"lances ilegais: {lances_ilegais}/{lances_modelo} ({resultado['taxa_lances_ilegais']:.2%})")
    if args.oponente == "stockfish":
        print("Elo estimado:", resultado["elo_estimado"] if resultado["elo_estimado"] is not None else "indefinido")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
