"""Gera traços de raciocínio VERIFICÁVEIS a partir do Stockfish.

O problema da medição anterior: o modelo nunca foi treinado a raciocinar, então
qualquer "análise" que ele produz é alucinação (colunas i-r, peças inventadas).
Medir isso não diz nada sobre raciocínio — diz que alucinação atrapalha.

Aqui os traços são construídos a partir de FATOS extraídos do motor:
  - quem está em xeque
  - se o lance dá mate
  - que peça é capturada
  - saldo material
  - casa do rei

Cada frase do traço é verificável por construção. Isso dá ao SFT um alvo honesto
e permite, depois, medir se o raciocínio aprendido tem relação com o acerto.
"""

from __future__ import annotations

import chess

VALOR = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

NOME_PT = {chess.PAWN: "peão", chess.KNIGHT: "cavalo", chess.BISHOP: "bispo",
           chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rei"}


def _cor(c: bool) -> str:
    return "brancas" if c == chess.WHITE else "pretas"


def _sinal_material(tabuleiro: chess.Board, cor: chess.Color) -> int:
    total = 0
    for _, peca in tabuleiro.piece_map().items():
        v = VALOR[peca.piece_type]
        total += v if peca.color == cor else -v
    return total


def gerar_traco(tabuleiro: chess.Board, lance: chess.Move) -> str:
    """Constrói um traço de raciocínio só com fatos verificáveis.

    Estrutura fixa em três passos, para o modelo aprender a *forma* de raciocinar
    antes de aprender o conteúdo:
      1. situação (xeque? material?)
      2. o que o lance faz (captura? xeque? mate? promove?)
      3. conclusão com o lance
    """
    lado = tabuleiro.turn
    linhas = []

    # 1. Situação, do ponto de vista de quem joga.
    if tabuleiro.is_check():
        linhas.append("Estou em xeque: preciso resolver a ameaça ao meu rei.")
    else:
        saldo = _sinal_material(tabuleiro, lado)
        if saldo > 0:
            linhas.append("Tenho vantagem material de %d ponto(s)." % saldo)
        elif saldo < 0:
            linhas.append("Estou atrás em material: %d ponto(s)." % saldo)
        else:
            linhas.append("O material está equilibrado.")

    # 2. O que o lance faz — tudo conferido no tabuleiro resultante.
    efeitos = []
    capturada = tabuleiro.piece_at(lance.to_square)
    if capturada is not None:
        efeitos.append("captura o %s" % NOME_PT[capturada.piece_type])
    if lance.promotion:
        efeitos.append("promove o peão a %s" % NOME_PT[lance.promotion])

    depois = tabuleiro.copy()
    depois.push(lance)

    if depois.is_checkmate():
        efeitos.append("dá xeque-mate")
    elif depois.is_check():
        efeitos.append("dá xeque")
    elif depois.is_check():
        pass

    # Ameaça concreta: peça adversária que passei a atacar.
    if not depois.is_checkmate():
        ameacadas = []
        for casa, peca in depois.piece_map().items():
            if peca.color != lado and depois.is_attacked_by(lado, casa):
                ameacadas.append((VALOR[peca.piece_type], NOME_PT[peca.piece_type]))
        ameacadas.sort(reverse=True)
        if ameacadas and not efeitos:
            efeitos.append("ataca o %s" % ameacadas[0][1])

    if not efeitos:
        efeitos.append("melhora a posição das minhas peças")

    linhas.append("O lance " + " e ".join(efeitos) + ".")

    # 3. Conclusão.
    linhas.append("LANCE: " + lance.uci())

    return "\n".join(linhas)


def traco_para_prompt(tabuleiro: chess.Board) -> str:
    """Prompt que PEDE o formato de raciocínio em três passos."""
    from .verificador import descrever_posicao
    lado = "brancas" if tabuleiro.turn == chess.WHITE else "pretas"
    return (
        "Você joga com as " + lado + " e é a sua vez.\n\n"
        "Posição (peças por casa):\n" + descrever_posicao(tabuleiro) + "\n\n"
        "FEN: " + tabuleiro.fen() + "\n\n"
        "Responda em três partes:\n"
        "1. A situação (xeque? vantagem material?)\n"
        "2. O que o seu lance faz (captura, xeque, mate, ameaça)\n"
        "3. A conclusão no formato: LANCE: <uci>\n\n"
        "Use apenas casas de a1 a h8."
    )


# --------------------------------------------------------------------------
# Traços com EXPLORAÇÃO DE CANDIDATOS (raciocínio de verdade)
# --------------------------------------------------------------------------

def _avaliar_candidatos(tabuleiro: chess.Board, motor, k: int = 3) -> list[tuple[chess.Move, float]]:
    """Avalia os k melhores lances segundo o Stockfish, em centipeões.

    É esta lista que dá ao traço o conteúdo de "como jogar": alternativas
    concretas com consequência medida, não uma narrativa de um lance só.
    """
    infos = motor.abrir().analyse(
        tabuleiro, chess.engine.Limit(depth=motor.profundidade), multipv=k)
    if isinstance(infos, dict):
        infos = [infos]

    saida = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        lance = pv[0]
        score = info["score"].pov(tabuleiro.turn)
        cp = float(score.score(mate_score=10000))
        saida.append((lance, cp))
    return saida


def _descrever_efeito(tabuleiro: chess.Board, lance: chess.Move) -> str:
    """Descreve em português o que o lance faz, conferido no tabuleiro."""
    partes = []
    capturada = tabuleiro.piece_at(lance.to_square)
    if capturada is not None:
        partes.append("captura %s" % NOME_PT[capturada.piece_type])
    if lance.promotion:
        partes.append("promove a %s" % NOME_PT[lance.promotion])

    depois = tabuleiro.copy()
    depois.push(lance)
    if depois.is_checkmate():
        partes.append("dá xeque-mate")
    elif depois.is_check():
        partes.append("dá xeque")
    else:
        ameacas = []
        for casa, peca in depois.piece_map().items():
            if peca.color != tabuleiro.turn and depois.is_attacked_by(tabuleiro.turn, casa):
                ameacas.append((VALOR[peca.piece_type], NOME_PT[peca.piece_type]))
        ameacas.sort(reverse=True)
        if ameacas:
            partes.append("ataca %s" % ameacas[0][1])

    if not partes:
        partes.append("reposiciona a peça")
    return " e ".join(partes)


def _fmt_cp(cp: float) -> str:
    """Faixa qualitativa, não número cru.

    O modelo não tem Stockfish na inferência: pedir que ele emita "+3.4" ensina
    a inventar números plausíveis. Faixas ("claramente favorável") são julgamentos
    que ele PODE derivar de fatos concretos — material, captura, xeque.
    """
    if cp >= 9000:
        return "mate a favor"
    if cp <= -9000:
        return "mate contra"
    if cp >= 300:
        return "claramente favorável"
    if cp >= 80:
        return "levemente favorável"
    if cp <= -300:
        return "claramente desfavorável"
    if cp <= -80:
        return "levemente desfavorável"
    return "equilibrada"


def gerar_traco_candidatos(tabuleiro: chess.Board, lance_correto: chess.Move,
                           motor, k: int = 3) -> str:
    """Traço com exploração de alternativas — o formato que ensina a jogar.

    Estrutura:
      1. Situação concreta (material, xeque)
      2. Candidatos com avaliação medida e efeito descrito
      3. Comparação: por que o escolhido é melhor
      4. Conclusão com o lance

    A comparação é o que faltava no template anterior: sem alternativas, o modelo
    aprende a narrar uma decisão já tomada, não a tomá-la.
    """
    lado = tabuleiro.turn
    linhas = ["Situação:"]

    if tabuleiro.is_check():
        linhas.append("- Estou em xeque.")
    saldo = _sinal_material(tabuleiro, lado)
    if saldo > 0:
        linhas.append("- Vantagem material de %d ponto(s)." % saldo)
    elif saldo < 0:
        linhas.append("- Desvantagem material de %d ponto(s)." % -saldo)
    else:
        linhas.append("- Material equilibrado.")

    candidatos = _avaliar_candidatos(tabuleiro, motor, k=k)
    # Garante o lance correto na lista, mesmo que o motor prefira outro.
    if all(m != lance_correto for m, _ in candidatos):
        cp = motor.avaliar_apos(tabuleiro, lance_correto)
        candidatos.append((lance_correto, cp))
    candidatos.sort(key=lambda par: -par[1])

    linhas.append("")
    linhas.append("Candidatos:")
    for mv, cp in candidatos[:k + 1]:
        efeito = _descrever_efeito(tabuleiro, mv)
        marca = "  <-- escolhido" if mv == lance_correto else ""
        linhas.append("- %s: %s, avaliação %s%s" % (mv.uci(), efeito, _fmt_cp(cp), marca))

    # Comparação explícita: o núcleo do raciocínio.
    melhor_cp = candidatos[0][1]
    cp_escolhido = next(cp for mv, cp in candidatos if mv == lance_correto)
    linhas.append("")
    if cp_escolhido >= melhor_cp:
        linhas.append("Comparação: %s é a melhor opção (%s), porque %s. "
                      "As alternativas deixam a posição %s."
                      % (lance_correto.uci(), _fmt_cp(cp_escolhido),
                         _descrever_efeito(tabuleiro, lance_correto),
                         _fmt_cp(candidatos[-1][1])))
    else:
        linhas.append("Comparação: %s é %s e mantém a posição, enquanto %s é %s."
                      % (lance_correto.uci(), _fmt_cp(cp_escolhido),
                         candidatos[0][0].uci(), _fmt_cp(melhor_cp)))

    linhas.append("")
    linhas.append("LANCE: " + lance_correto.uci())
    return "\n".join(linhas)


PROMPT_CANDIDATOS = (
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


def prompt_candidatos(tabuleiro: chess.Board) -> str:
    from .verificador import descrever_posicao
    lado = "brancas" if tabuleiro.turn == chess.WHITE else "pretas"
    return PROMPT_CANDIDATOS.format(lado=lado,
                                    posicao=descrever_posicao(tabuleiro),
                                    fen=tabuleiro.fen())


# ---------------------------------------------------------------------------
# Varredura de lances forçados
# ---------------------------------------------------------------------------
#
# Medição que motiva este formato (n=3000, scripts/diagnostico_forcados.py):
#
#   lances legais por posição ........ 30.5
#   lances FORÇADOS (xeque/captura) ... 3.9
#   solução é forçada ............... 80.2%
#   recall listando todos os forçados  80.2%
#   recall aleatório (3 de 31) ...... 9.8%
#
# O modelo listava 3 candidatos e acertava incluir a solução em 12,83% —
# praticamente o aleatório. Varrer o que é forçado custa o MESMO orçamento de
# tokens (~4 candidatos) e cobre 80,2%. O gargalo não é o tamanho da lista, é
# a ausência de um procedimento de busca.

MAX_FORCADOS = 8   # teto: 79,9% de recall com 3,78 candidatos (ver out/teto-forcados.json)


def lances_forcados(tabuleiro: chess.Board,
                    maximo: int = MAX_FORCADOS) -> list[chess.Move]:
    """Xeques e capturas, ordenados por urgência e valor.

    Ordem: xeques primeiro (mais forçados), depois capturas por valor da peça
    capturada. Promoções entram junto das capturas.
    """
    xeques, capturas = [], []
    for mv in tabuleiro.legal_moves:
        if tabuleiro.gives_check(mv):
            xeques.append(mv)
        elif tabuleiro.is_capture(mv) or mv.promotion:
            valor = 0
            alvo = tabuleiro.piece_at(mv.to_square)
            if alvo is not None:
                valor = VALOR[alvo.piece_type]
            capturas.append((valor, mv))

    capturas.sort(key=lambda par: -par[0])
    ordenados = xeques + [mv for _, mv in capturas]
    return ordenados[:maximo]


def _em_xeque(tabuleiro: chess.Board) -> str:
    return "sim" if tabuleiro.is_check() else "não"


def gerar_traco_forcados(tabuleiro: chess.Board, lance_correto: chess.Move,
                         motor=None) -> str:
    """Traço que executa o PROCEDIMENTO de busca, não narra um lance escolhido.

    Estrutura:
      Situação   — material e xeque (fatos do tabuleiro)
      Xeques     — todos os xeques disponíveis, com o efeito de cada um
      Capturas   — todas as capturas, com o efeito de cada uma
      Comparação — por que o escolhido é o melhor entre os forçados
      LANCE      — o lance

    Se o lance correto não for forçado (≈20% dos casos), ele é acrescentado
    explicitamente para o modelo ver que às vezes a resposta é um lance quieto.
    """
    forcados = lances_forcados(tabuleiro)
    if lance_correto not in forcados:
        forcados = forcados + [lance_correto]

    xeques = [m for m in forcados if tabuleiro.gives_check(m)]
    outros = [m for m in forcados if m not in xeques]

    linhas = ["Situação:"]
    sinal = _sinal_material(tabuleiro, tabuleiro.turn)
    if sinal > 0:
        linhas.append("- Vantagem material de %d ponto(s)." % sinal)
    elif sinal < 0:
        linhas.append("- Desvantagem material de %d ponto(s)." % -sinal)
    else:
        linhas.append("- Material equilibrado.")
    linhas.append("- Em xeque: %s." % _em_xeque(tabuleiro))
    linhas.append("")

    if xeques:
        linhas.append("Xeques:")
        for mv in xeques:
            marca = "  <-- escolhido" if mv == lance_correto else ""
            linhas.append("- %s: %s%s" % (mv.uci(), _descrever_efeito(tabuleiro, mv), marca))
    else:
        linhas.append("Xeques: nenhum.")
    linhas.append("")

    if outros:
        linhas.append("Capturas e promoções:")
        for mv in outros:
            marca = "  <-- escolhido" if mv == lance_correto else ""
            linhas.append("- %s: %s%s" % (mv.uci(), _descrever_efeito(tabuleiro, mv), marca))
    else:
        linhas.append("Capturas e promoções: nenhuma.")
    linhas.append("")

    # A comparação é o passo que ensina a decidir entre os forçados.
    efeito = _descrever_efeito(tabuleiro, lance_correto)
    if tabuleiro.gives_check(lance_correto):
        motivo = "é xeque, então força a resposta do oponente"
    elif tabuleiro.is_capture(lance_correto):
        motivo = "ganha material"
    elif lance_correto.promotion:
        motivo = "promove o peão"
    else:
        motivo = "é o único que não deixa o oponente com iniciativa"

    linhas.append("Comparação: entre os lances forçados, %s é o melhor porque %s "
                  "e %s." % (lance_correto.uci(), motivo, efeito))
    linhas.append("")
    linhas.append("LANCE: %s" % lance_correto.uci())
    return "\n".join(linhas)


PROMPT_FORCADOS = (
    "Você joga com as {lado} e é a sua vez.\n\n"
    "Posição (peças por casa):\n{posicao}\n\n"
    "FEN: {fen}\n\n"
    "Procure nesta ordem, sem pular etapas:\n"
    "1. Situação: material e se está em xeque\n"
    "2. Xeques: liste TODOS os xeques disponíveis e o que cada um faz\n"
    "3. Capturas e promoções: liste TODAS e o que cada uma faz\n"
    "4. Comparação: entre os lances forçados, qual é o melhor e por quê\n"
    "5. LANCE: <uci>\n\n"
    "Quase sempre a resposta é um lance forçado (xeque ou captura).\n"
    "Use apenas casas de a1 a h8."
)


def prompt_forcados(tabuleiro: chess.Board) -> str:
    """Prompt que pede o procedimento de varredura, não a resposta direta."""
    lado = "brancas" if tabuleiro.turn else "pretas"
    from .verificador import descrever_posicao
    return PROMPT_FORCADOS.format(
        lado=lado, posicao=descrever_posicao(tabuleiro), fen=tabuleiro.fen())
