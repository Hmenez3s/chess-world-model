# chess-world-model

Uma LLM pequena (Qwen2.5-1.5B) treinada para entender o tabuleiro de xadrez antes de
escolher o lance. Treinada com LoRA numa única **RTX 3060 de 12 GB**, sem nada treinado do
zero.

## Início vs fim

Split de teste sem vazamento (divisão por partida), 600 puzzles do Lichess, mesma amostra
para os dois lados.

| | Início: Qwen2.5-1.5B puro | Fim: modelo de mundo + lance verificado |
|---|---|---|
| **Elo de puzzle (Lichess)** | ≤ 825 (0 acertos) | **1460** [1408; 1509] |
| Puzzles resolvidos | 0,00% | **16,00%** [13,28; 19,15] |
| Lance legal | 3,33% | 99,17% |
| Desenha o tabuleiro após o lance | não consegue | 93,83% exato |
| Partidas vs jogador aleatório (V/E/D) | 0/6/0 | 1/5/0 |
| Partidas vs Stockfish `UCI_Elo=1320` (V/E/D) | 0/1/7 | 0/0/8 |
| Lances ilegais em partida | 96,31% | 5,21% |

O **Elo de puzzle** é estimado pelos ratings do Lichess dos puzzles que o modelo resolve
(máxima verossimilhança, IC95 por bootstrap). Quando o modelo não resolve nenhum, o Elo
não tem estimativa pontual; o limite mostrado é o maior Elo compatível com 0 acertos.

**Como ler as partidas.** O Qwen puro joga lance ilegal em ~96% das vezes; nesses casos o
lance é substituído por um lance legal aleatório. Então o placar do início é, na prática, o
de um jogador aleatório (inclusive o empate contra o Stockfish). O modelo final joga os
próprios lances (5,21% ilegais) e mesmo assim perde para o Stockfish no nível mais fraco que
ele aceita (1320). O ganho está na leitura do tabuleiro e na escolha do lance em posições
táticas; jogar uma partida inteira exige manter um plano por dezenas de lances, e o treino
não cobriu isso.

## Como chegou lá

| Etapa | Puzzles resolvidos | Elo de puzzle |
|---|---|---|
| Qwen2.5-1.5B puro | 0,00% | ≤ 825 (0 acertos) |
| + SFT só com o lance | 4,67% | 1190 [1109; 1254] |
| + SFT + GRPO com Stockfish | 6,17% | 1246 [1180; 1302] |
| Modelo de mundo, 30k posições | 8,00% | 1300 [1237; 1353] |
| Modelo de mundo, 70k posições | 13,67% | 1421 [1370; 1475] |
| **+ lance verificado (final)** | 16,00% | **1460** [1408; 1509] |
| + GRPO curto (não significativo) | 16,83% | 1473 [1421; 1518] |

Treinar só o lance certo (SFT) e depois reforçar com RL (GRPO) quase não mexeu no Elo: o
modelo aprendia a **forma** da resposta, não o jogo. O salto veio quando o treino passou a
pedir o **modelo de mundo**.

## O método: modelo de mundo

Adaptado de *Mastering Board Games by External and Internal Planning with Language Models*
(DeepMind, [arXiv 2412.12119](https://arxiv.org/abs/2412.12119)). Em vez de pedir só o
lance, cada exemplo pede ao modelo que escreva o que acontece no jogo. Ele recebe:

```
FEN anterior: r4rk1/2q3p1/p1bbp3/1p3p2/4NP2/5Q1P/PPPB2P1/4RRK1 w - - 0 20
Lance jogado: e4d6
```

e tem de escrever:

```
estado:
8 r . . . . r k .
7 . . q . . . p .
6 p . b N p . . .      ← o cavalo saiu de e4 e capturou o bispo em d6
5 . p . . . p . .
4 . . . . . P . .      ← e4 ficou vazia
3 . . . . . Q . P
2 P P P B . . P .
1 . . . . R R K .
  a b c d e f g h
vez: pretas
legais: a6a5 a8a7 ... c6f3 ... c7d6 ... (35 lances)
top5: c6f3:85 c7d6:40 a8d8:31 f8d8:31 f8f6:25
melhor: c6f3
```

1. **O tabuleiro depois do lance.** Maiúscula é peça branca, minúscula é preta.
2. **Os lances legais** da nova posição.
3. **O valor de cada um dos 5 melhores lances**, como chance de vitória (rótulo do
   Stockfish, usado só para gerar os dados de treino).
4. **O melhor lance.**

O melhor lance do exemplo, `c6f3` (o bispo captura a dama), só existe porque o cavalo
**saiu de e4** e abriu a diagonal. Um modelo que não atualiza o tabuleiro não vê esse
lance. Por isso ensinar o estado melhora a escolha.

**Lance verificado.** Como o modelo escreve a própria lista de lances legais e o top-5,
quando o `melhor:` sai ilegal a resposta ainda tem candidatos válidos. O lance final passa a
ser o do modelo se for legal; senão, o primeiro legal do top-5 dele; senão, o primeiro da
lista dele. Não entra informação externa. Na partida, isso reduz os lances ilegais de 35,86%
para 5,21%.

### Detalhe por camada

| Modelo | Tabuleiro exato | Cor | Lances legais (F1) | Melhor lance = solução |
|---|---|---|---|---|
| Qwen2.5-1.5B puro | não pede | — | — | 0,00% [0,00; 0,64] |
| + SFT só com o lance | não pede | — | — | 4,67% [3,25; 6,66] |
| + SFT + GRPO com Stockfish | não pede | — | — | 6,17% [4,51; 8,38] |
| Modelo de mundo, 30k posições | 86,33% | 99,71% | 0,00% (0,555) | 8,00% [6,09; 10,45] |
| Modelo de mundo, 70k posições | 93,83% | 99,90% | 5,17% (0,799) | 13,67% [11,15; 16,65] |
| **Modelo de mundo, 70k + lance verificado** | 93,83% | 99,90% | 5,17% (0,799) | 16,00% [13,28; 19,15] |
| + GRPO curto sobre o modelo (não significativo) | 93,67% | 99,90% | 5,00% (0,797) | 16,83% [14,05; 20,04] |

## Generalização

Mesma bateria no modelo do início e no do fim. Lig 4 é um jogo que o modelo nunca viu (regras
no prompt); GSM8K mede se ele esqueceu matemática.

| Tarefa | Início (Qwen puro) | Fim (treinado) | Δ |
|---|---|---|---|
| Lig 4: vencer agora | 15,00% | 20,00% | +5,0 pp |
| Lig 4: bloquear a ameaça | 17,00% | 9,00% | -8,0 pp |
| Lig 4: total (acaso: 15,3%) | 16,00% | 14,50% | -1,5 pp |
| GSM8K (matemática) | 51,00% | 20,50% | -30,5 pp |

- **Não transferiu para outro jogo.** No Lig 4 o modelo final ficou no acaso. Aprender a ler
  o tabuleiro de xadrez não o ensinou a ler outro tabuleiro.
- **Esqueceu matemática.** O GSM8K caiu para menos da metade. O modelo passa a responder
  perguntas de matemática no formato de xadrez (`estado:` seguido de um "tabuleiro"). É o
  custo de um fine-tune estreito; mistura com dados gerais não foi testada aqui.

**Memorização:** com as posições espelhadas (mesma tática, coordenadas trocadas) o acerto não
cai, então o modelo não decorou os puzzles.

| Modelo | Original | Espelhado | Queda | Descartados |
|---|---|---|---|---|
| SFT só com o lance | 4,67% | 4,67% | +0,0 pp | 0 |
| Modelo de mundo | 16,00% | 17,33% | -1,3 pp | 0 |

## Custo

| Etapa | Exemplos | Passos | Tempo | VRAM pico |
|---|---|---|---|---|
| Modelo de mundo, 30k | 30000 | 1875 | 148 min | 5,41 GB |
| Modelo de mundo, +40k (continuação) | 71704 | 2200 | 174 min | 5,41 GB |

## Limites

- **Escala pequena.** São 70 mil posições e LoRA num modelo de 1,5B. Os trabalhos de
  referência usam centenas de milhões de posições e treinam o transformer do zero.
- **O Elo de puzzle é uma aproximação.** Ele mede a força implícita na taxa de acerto em
  posições táticas, não a força numa partida. Em partida o modelo perde para o Stockfish 1320.
- **Ainda não há busca interna.** O passo seguinte da literatura é o modelo escrever a própria
  árvore ("eu jogo, ele responde, eu avalio") e treinar nesses traços. Não foi feito.
- **Generalização negativa.** Não houve transferência para o Lig 4 e houve esquecimento de
  matemática.

## Reproduzir

```bash
pip install -e ".[train]"
export STOCKFISH_PATH=~/bin/stockfish

# 1. splits sem vazamento a partir do Lichess open database
python scripts/preparar_splits.py --limite 0

# 2. dados do modelo de mundo (só o split de treino), rotulados pelo Stockfish
python scripts/gerar_mav.py --posicoes 32000 --processos 12 --saida data/mav-32k.jsonl
python scripts/gerar_mav.py --posicoes 40000 --pular 32000 --processos 12 --saida data/mav-72k.jsonl
cat data/mav-32k.jsonl data/mav-72k.jsonl > data/mav-uniao.jsonl

# 3. treino (RTX 3060: ~2,5 h + ~2,9 h)
python scripts/treinar_sft_cot2.py --adapter-base none --tracos data/mav-32k.jsonl \
       --amostras 30000 --lote 8 --acum 2 --lr 2e-4 --max-length 512 --saida out/mav-30k
python scripts/treinar_sft_cot2.py --adapter-base out/mav-30k --tracos data/mav-uniao.jsonl \
       --lote 8 --acum 2 --lr 1e-4 --max-length 512 --max-passos 2200 --saida out/mav-final

# 4. início vs fim no mesmo teste
python scripts/avaliar_mav.py --formato direto --puzzles 600 --saida out/aval-teste-base.json
python scripts/avaliar_mav.py --adapter out/mav-final --puzzles 600 --verificar \
       --saida out/aval-mav-verificado.json
python scripts/elo_puzzle.py out/aval-teste-base.json out/aval-mav-verificado.json
python scripts/jogar_partidas.py --adapter out/mav-final --oponente stockfish --elo 1320 \
       --partidas 8 --verificar --saida out/partidas-stockfish-verificado.json

# 5. README gerado a partir dos JSONs em out/ (falha se faltar algum)
python scripts/gerar_readme.py
```

O adapter treinado (LoRA r=64, 275 MB) está na [Release](../../releases):

```bash
tar xzf adapter-mav-final.tar.gz -C out/mav-final
python scripts/avaliar_mav.py --adapter out/mav-final --puzzles 50 --verificar
```

## Outros métodos testados

| Método | Acerto em puzzle | Observação |
|---|---|---|
| SFT direto | 4,67% (n=600) | aprende a forma do lance |
| SFT + GRPO com Stockfish | 6,17% (n=600) | recompensa densa |
| SFT + raciocínio em texto | 9,17% (n=600) | amostra anterior ao split de teste |
| Raciocínio + GRPO | 9,67% (n=600) | amostra anterior ao split de teste |
| Varredura de xeques e capturas | 9,17% (n=600) | amostra anterior ao split de teste |

- **GRPO direto** melhora a legalidade, mas não o acerto: o modelo otimiza a parte fácil da
  recompensa.
- **Raciocínio em texto** ("situação, candidatos, comparação") dobra o acerto dentro do
  mesmo modelo, mas não supera o SFT direto: o modelo escreve uma comparação plausível sem
  ler o tabuleiro.
- **Ensinar a varrer xeques e capturas** coloca a solução mais vezes na lista de candidatos,
  mas piora a escolha final.

Todos esses métodos atacam a escolha. O modelo de mundo ataca a leitura do tabuleiro, e só
ele moveu o Elo.

## Dados e licenças

Puzzles: [Lichess open database](https://database.lichess.org/#puzzles) (CC0). Avaliações
geradas com [Stockfish](https://stockfishchess.org/) (GPLv3), usado só para rotular dados e
medir; não é distribuído aqui. Código: MIT.
