# chess-world-model — uma LLM aprendendo o modelo de mundo do xadrez

**Tese:** um LLM pequeno não falha no xadrez por falta de tática, e sim por não conseguir
reconstruir o tabuleiro a partir do texto. Treinar só a resposta ("qual o melhor lance?")
ensina a forma da resposta, não o jogo. Este repositório mede isso camada por camada e
treina a camada que falta: **o modelo de mundo** — o tabuleiro depois do lance, os lances
legais e o valor de cada lance.

Xadrez aqui é laboratório: é barato de verificar, tem gabarito objetivo e separa
"raciocínio" de "conhecimento". O que se busca não é um enxadrista, e sim entender como
uma LLM aprende a **pensar à frente** — e se isso generaliza.

## O problema, medido

Acurácia em puzzle mistura cinco capacidades num número só. Sondas isoladas (mesma
representação de tabuleiro em todas, n=150, resposta gulosa) mostram onde o modelo quebra:

| Nível | Base | Depois do SFT | Acaso |
|---|---|---|---|
| Tipo da peça numa casa | 79,33% | 90,00% | ~17% |
| **Cor da peça numa casa** | **48,67%** | **58,67%** | **50%** |
| Contagem de peões | 8,67% | 2,00% | ~17% |
| Quem tem mais material | 33,33% | 36,00% | 33% |
| Rei em xeque | 0,67% | 17,33% | 50% |
| Julgar se um lance é legal | 58,67% | 57,33% | 50% |
| **Enumerar os lances legais de uma peça** | **0,83%** | **0,83%** | ~0% |

O prompt diz literalmente `a3: peão branco`. O modelo lê o tipo da peça e **não lê a cor** —
48,67%, exatamente o acaso. Sem cor, metade do tabuleiro é desconhecida. E a enumeração de
lances legais fica em 0,83%: ele não gera o espaço de ações.

Isso explica o que veio antes: SFT direto estaciona em 4,67% no split de teste, GRPO com
recompensa densa chega a 6,17% e não há ganho significativo. Recompensar a escolha não
conserta a leitura.

## O método: modelo de mundo (formato MAV)

Adaptado de *Mastering Board Games by External and Internal Planning with Language Models*
(DeepMind, [arXiv 2412.12119](https://arxiv.org/abs/2412.12119)), que treina um transformer
de 1B com este formato e chega a ~2.820 Elo estimado sem busca externa.

Cada exemplo de treino é um texto único. A partir do FEN anterior e do lance jogado, o
modelo escreve:

```
FEN anterior: r4rk1/2q3p1/p1bbp3/1p3p2/4NP2/5Q1P/PPPB2P1/4RRK1 w - - 0 20
Lance jogado: e4d6
```

```
estado:
8 r . . . . r k .
7 . . q . . . p .
6 p . b N p . . .      ← o cavalo saiu de e4 e capturou o bispo em d6
5 . p . . . p . .
4 . . . . . P . .
3 . . . . . Q . P
2 P P P B . . P .
1 . . . . R R K .
  a b c d e f g h
vez: pretas
legais: a6a5 a8a7 ... c6f3 ... c7d6 ... (35 lances)
top5: c6f3:85 c7d6:40 a8d8:31 f8d8:31 f8f6:25
melhor: c6f3
```

O `top5` traz, para cada lance, a chance de vitória estimada pelo Stockfish. Isso é o que
diferencia este treino: o modelo não aprende apenas qual lance foi escolhido, aprende a
**avaliar** os candidatos. No ChessBench ([arXiv 2402.04494](https://arxiv.org/abs/2402.04494)),
prever valor por lance acerta 83,3% dos puzzles, contra 65,7% de prever só o melhor lance,
no mesmo modelo.

A grade 8×8 usa maiúscula para peça branca e minúscula para preta: a cor fica explícita e
cada casa ocupa uma posição fixa, sem depender de como o tokenizador BPE quebra o FEN.

### Lance verificado: o modelo confere a própria saída

O formato tem uma propriedade útil que o treino sozinho não explora: **o modelo escreve a
lista de lances legais e o top-5 antes de concluir**. Quando o `melhor:` sai ilegal, a saída
dele ainda contém candidatos válidos na mesma resposta.

`lance_verificado()` usa isso: aceita o lance do modelo se for legal; senão pega o primeiro
lance legal do `top5:`; senão o primeiro da lista `legais:`. Não dá informação nova ao
modelo — só impede que ele jogue fora da própria lista. O efeito é grande:

| | Lance legal | Acerto |
|---|---|---|
| Saída crua do modelo | 79,67% | 13,67% |
| **Com lance verificado** | **99,17%** | **16,00%** |

Em partida completa a diferença é maior ainda: os 35,86% de lances ilegais caem para perto de
zero, porque toda posição passa pela mesma conferência.

## Resultados

Split de teste, sem vazamento (divisão por `GameId`, 0 jogos compartilhados), n=600,
seed 42, decodificação gulosa. A linha de base foi medida **neste mesmo split**.

### Camadas

{{TABELA_CAMADAS}}

### Elo

Elo de puzzle é estimado por máxima verossimilhança logística sobre os ratings do Lichess
dos puzzles do teste (IC95 por bootstrap de itens). É um proxy: mede a força implícita na
taxa de acerto, não a força em partida.

{{TABELA_ELO}}

Partidas completas, cores alternadas, lances gerados em lote:

{{TABELA_PARTIDAS}}

O contraste entre as duas tabelas é o resultado mais instrutivo do repositório:

- **Nos puzzles, o modelo é decente**: 16,00% de acerto e Elo de puzzle **1460**, acima do SFT
  direto (1190) e do GRPO com Stockfish (1246), com intervalos que não se sobrepõem.
- **Em partida completa, ele é fraco**: ganha ou empata contra um jogador aleatório (0,583 de
  pontuação), mas perde todas as 8 partidas para um Stockfish limitado a 1320. Com 0% de
  pontuação o Elo é matematicamente indefinido; o limite superior compatível é **≤ 1193**.
  Não é contradição: puzzle é uma posição tática isolada; partida completa exige manter o
  estado por dezenas de lances, e o treino nunca cobriu essa sequência.
- **O lance verificado resolveu a legalidade, não a força.** Sem ele, 35,86% dos lances em
  partida eram ilegais; com ele, 5,21%. Mas o placar contra o Stockfish não melhorou: o modelo
  passou a jogar **legal e fraco**, em vez de ilegal. Ou seja, o gargalo de partida não era
  sintaxe, era estratégia — exatamente o que o treino de puzzle não ensina.

Ou seja: o modelo de mundo melhora a **leitura** e a **escolha local**, e ainda não sustenta
**jogo completo**. As duas coisas são medidas separadamente de propósito, e a distância entre
elas é o resultado.

### Generalização: o treino ensinou xadrez ou ensinou a pensar?

Mesma bateria no modelo base e no treinado. Lig 4 é um jogo que o modelo nunca viu, com as
regras no prompt e verificação em Python; GSM8K mede esquecimento.

{{TABELA_GENERALIZACAO}}

A leitura honesta dos dois lados:

- **Não houve transferência para um jogo novo.** No Lig 4 o modelo treinado ficou no acaso
  nas duas variantes (vencer e bloquear). Aprender a ler *um* tabuleiro não o ensinou a ler
  outro: o formato de representação faz parte do que foi aprendido, não é um raciocínio
  transferível.
- **Houve esquecimento severo de matemática.** O GSM8K caiu para menos da metade. Inspecionando
  as respostas, o modelo treinado passa a responder perguntas de matemática **no formato de
  xadrez** — escreve `estado:` e tenta desenhar um tabuleiro com "bolts of fiber". Não é só
  perda de conteúdo: o formato de saída dominou a distribuição.

Isso é o custo medido de fine-tune estreito com muitas sequências longas num único formato. A
troca é explícita: **+231 Elo de puzzle dentro do domínio, −30 pp de matemática fora dele.**
Quem quiser as duas coisas precisa de mistura de dados ou regularização — nenhuma das duas foi
usada aqui.

### Memorização

Espelho vertical com troca de cores: mesma tática, coordenadas diferentes. Queda grande
indicaria decoreba.

{{TABELA_ESPELHO}}

### Custo

Treinado do zero? Não. LoRA r=64 sobre `Qwen/Qwen2.5-1.5B-Instruct`, numa única
**RTX 3060 de 12 GB** (`CUDA_VISIBLE_DEVICES=1`).

{{TABELA_ESCALA}}

## Reproduzir

```bash
pip install -e .                     # python-chess, torch, transformers, peft, trl, datasets
export STOCKFISH_PATH=~/bin/stockfish

# 1. splits sem vazamento a partir do Lichess open database
python scripts/preparar_splits.py --limite 0
python scripts/conferir_splits.py

# 2. dados no formato MAV (só o split de treino), rotulados pelo Stockfish
python scripts/gerar_mav.py --posicoes 32000 --processos 12 --saida data/mav-32k.jsonl
python scripts/gerar_mav.py --posicoes 40000 --pular 32000 --processos 12 \
       --saida data/mav-72k.jsonl
cat data/mav-32k.jsonl data/mav-72k.jsonl > data/mav-uniao.jsonl

# 3. treino na 3060: ~2,6 h para o primeiro estágio e ~2,9 h para a continuação
python scripts/treinar_sft_cot2.py --adapter-base none --tracos data/mav-32k.jsonl \
       --amostras 30000 --lote 8 --acum 2 --lr 2e-4 --max-length 512 --saida out/mav-30k
python scripts/treinar_sft_cot2.py --adapter-base out/mav-30k --tracos data/mav-uniao.jsonl \
       --lote 8 --acum 2 --lr 1e-4 --max-length 512 --max-passos 2200 --saida out/mav-final

# 4. avaliação (--verificar = confere o lance contra a posicao, usando a
#    propria lista de legais e o top5 que o modelo escreveu)
python scripts/avaliar_mav.py --adapter out/mav-final --puzzles 600 \
       --verificar --saida out/aval-mav-verificado.json
python scripts/avaliar_mav.py --adapter out/mav-final --puzzles 600 \
       --verificar --espelho --saida out/espelho-mav-verificado.json
python scripts/elo_puzzle.py out/aval-mav-verificado.json out/aval-teste-sft.json
python scripts/jogar_partidas.py --adapter out/mav-final --oponente stockfish \
       --elo 1320 --partidas 8 --verificar --saida out/partidas-stockfish-verificado.json
python scripts/generalizacao.py --adapter out/mav-final --saida out/generalizacao-treinado.json
python scripts/generalizacao.py --saida out/generalizacao-base.json

# 5. README gerado da evidência (falha se faltar algum JSON)
python scripts/gerar_readme.py
```

## Limites

- **LoRA num modelo de 1,5B, com 70 mil exemplos.** Os trabalhos da área usam centenas de
  milhões de posições e treinam o transformer do zero. Aqui a escala é ordens de grandeza
  menor; o resultado mede se a receita move a agulha nesse orçamento, não o teto dela.
- **Sem busca interna ainda.** O próximo passo da literatura é ensinar o modelo a escrever a
  própria árvore ("eu jogo, ele responde, eu avalio") e treinar com traços de busca
  (MAV-IS, Searchformer, Stream of Search). Não foi feito aqui.
- **Elo de puzzle é proxy.** Vem da taxa de acerto sobre os ratings do Lichess; não é Elo de
  torneio. O Elo de partida só é reportado quando o placar não é 0% nem 100%.
- **Generalização medida em três tarefas.** Lig 4 e GSM8K são sondas, não um veredito sobre
  raciocínio estratégico geral. O resultado encontrado é negativo (sem transferência, com
  esquecimento) e está reportado como tal.
- **O formato de treino domina a saída.** O modelo treinado responde fora do domínio no formato
  de xadrez. Mitigar isso (mistura de dados gerais, regularização, menos passos) é trabalho
  seguinte, não foi feito aqui.
- **Amostra de teste de 600 puzzles** para as camadas e o Elo, com IC95 reportado; diferenças
  dentro do intervalo não devem ser lidas como ganho.

## Outros métodos testados (fora do foco)

Todos medidos no mesmo pipeline. Ficam registrados porque cada um ensinou algo sobre o
problema — e porque os resultados negativos são parte do resultado.

{{TABELA_OUTROS}}

O que cada um mostrou:

- **GRPO ingênuo faz reward hacking na parte densa da recompensa.** Legalidade sobe a 82,5%
  enquanto o acerto cai a 3,5%, abaixo do SFT: o modelo otimiza o que é fácil medir.
- **Currículo por rating conserta o colapso de variância** (grupos com variância zero caem de
  86% para 35%), e a recompensa contínua do Stockfish reduz para 11–26%. Mas isso é sinal de
  treino, não acurácia final: a ablação dá +0,34 pp, z = +0,20.
- **Raciocínio estruturado dobra o acerto dentro do mesmo modelo** (9,67% contra 4,17%
  forçado a responder direto, z = +3,75), mas não supera o SFT direto no agregado.
- **Busca e seleção competem pelo mesmo espaço.** Ensinar o modelo a varrer xeques e capturas
  subiu o recall dos candidatos de 12,83% para 20,83%, mas derrubou a seleção de 74,03% para
  40,80%: acerto final 8,50%, sem ganho. Com o lance certo na lista ele acerta 74%; a lista só
  continha o certo em 12,83% — quase o acaso (9,8% para 3 de 31 lances legais). O gargalo é a
  busca, e foi isso que motivou o modelo de mundo.
- **Nada disso melhorou a leitura do tabuleiro.** Foi o que a escada mostrou, e é o motivo de
  o método principal deste repositório atacar estado e regras em vez da resposta.

## Adapter

Os pesos treinados (LoRA r=64, 293 MB) estão na
[Release](../../releases) deste repositório como `adapter-mav-final.tar.gz`, já que o
Hugging Face exige token de escrita. Para usar:

```bash
tar xzf adapter-mav-final.tar.gz -C out/mav-final
python scripts/avaliar_mav.py --adapter out/mav-final --puzzles 50 --verificar
```

## Dados e licenças

Puzzles: [Lichess open database](https://database.lichess.org/#puzzles) (CC0).
Avaliações de posição geradas com [Stockfish](https://stockfishchess.org/) (GPLv3), usado
apenas para rotular dados e medir; não é distribuído aqui.
