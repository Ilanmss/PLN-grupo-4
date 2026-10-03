# Relatório de resultados: classificação de clareza de respostas do e-SIC

Gerado automaticamente por `gerar_relatorio.py` em 03/10/2026 16:41.

> ⏳ **BERTimbau + TAPT, fine-tuning (2 épocas): validação cruzada incompleta (1 de 5 folds).** Nos folds prontos (1): F1 0,4546, contra 0,4536 do BERTimbau e 0,4412 do baseline. Fica fora do ranking, que exige os 5 folds (decisão e análise em `EXPERIMENTOS.md`).
>
> ⏳ **Albertina-100m PTBR (DeBERTa), fine-tuning (2 épocas): validação cruzada incompleta (1 de 5 folds).** Nos folds prontos (1): F1 0,4370, contra 0,4536 do BERTimbau e 0,4412 do baseline. Fica fora do ranking, que exige os 5 folds (decisão e análise em `EXPERIMENTOS.md`).

## 1. Resumo

Validação cruzada agrupada por texto com 5 fold(s); média ± desvio padrão entre folds.

| Modelo | F1-macro (validação cruzada) | Acurácia (validação cruzada) | F1 armazenado (divisão única 80/20) | Ganho de F1 sobre o baseline |
|---|---|---|---|---|
| Baseline classe majoritária | 0,1693 ± 0,0021 | 0,3403 ± 0,0057 | 0,1703 | — |
| Baseline TF-IDF + regressão logística | 0,4521 ± 0,0061 | 0,4541 ± 0,0065 | 0,4648 | referência |
| **Melhor modelo com BERT:** Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | **0,4686 ± 0,0056** | **0,4682 ± 0,0059** | — | **+1,7 pts** |
| **Melhor modelo com outros métodos:** Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | **0,4597 ± 0,0040** | **0,4589 ± 0,0048** | — | **+0,8 pts** |

- **F1 armazenado:** valor registrado em `modelos/baselines/resultados_baselines.txt`, obtido pelo `baselines.py` numa divisão aleatória 80/20 (semente 123, 4019 textos de teste). Recarregando os modelos `.joblib` armazenados nessa divisão, obtive F1 0,1703 (acurácia 0,3431) para a classe majoritária e F1 0,4648 (acurácia 0,4675) para o TF-IDF + regressão logística, idênticos aos registrados.
- **Ganho:** diferença de F1-macro médio para o baseline TF-IDF + regressão logística, nos mesmos folds.
- A nota do EP usa a **acurácia** no teste. Como as classes são quase balanceadas, F1-macro e acurácia andam juntos. Pela acurácia, o melhor com BERT seria Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada) (acurácia 0,4706 ± 0,0036, F1 0,4663 ± 0,0040). Pela acurácia, o melhor sem BERT seria Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada) (acurácia 0,4596 ± 0,0049, F1 0,4560 ± 0,0038). O ajuste de viés favorece a classe c234, o que aumenta o F1-macro mas pode reduzir um pouco a acurácia.

## 2. Protocolo de avaliação

- **Validação cruzada agrupada, 5 folds** (`GroupKFold`, embaralhado, semente 42), agrupando textos idênticos (após normalizar espaços). Nenhuma cópia de um texto fica no treino e na avaliação ao mesmo tempo. Sem o agrupamento, templates repetidos, como o do MTb que aparece 81 vezes, inflam o resultado: o mesmo LinearSVC dá F1 0,4595 com `StratifiedKFold` e 0,4544 com `GroupKFold`.
- **Os mesmos folds para todos os modelos** (`src/divisoes.py`). Cada modelo é treinado 5 vezes, cada vez sem um fold, e prevê o fold que ficou de fora. As predições resultantes são chamadas de *out-of-fold*.
- **Ensembles:** combinam as probabilidades out-of-fold. Pesos e viés por classe são ajustados nos outros folds e aplicados ao fold avaliado (*cross-fitting*), para que nenhum parâmetro seja escolhido olhando os dados em que é medido.
- **Seleção:** o melhor de cada categoria é o de maior F1-macro médio. Como vários candidatos são comparados nos mesmos folds, o valor do vencedor pode estar levemente otimista, em fração do desvio padrão.
- O conjunto de teste não rotulado **não foi usado** em nenhuma etapa.

## 3. Baselines

| Baseline | F1 armazenado (divisão 80/20) | F1 validação cruzada | Acurácia validação cruzada |
|---|---|---|---|
| Classe majoritária | 0,1703 | 0,1693 ± 0,0021 | 0,3403 ± 0,0057 |
| TF-IDF + regressão logística (baseline oficial) | 0,4648 | 0,4521 ± 0,0061 | 0,4541 ± 0,0065 |

O TF-IDF + regressão logística tem F1 menor na validação cruzada agrupada do que na divisão aleatória, porque nela não consegue "decorar" templates repetidos. O mesmo vale para os demais modelos, então as comparações entre modelos são feitas sempre na mesma avaliação.

## 4. Melhor modelo usando BERT

**Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés)**: F1-macro **0,4686 ± 0,0056**, acurácia **0,4682 ± 0,0059** (+1,7 pts sobre o baseline).

Pesos finais (BERTimbau fine-tuning (2 épocas): 0,4787, TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC: 0,2978, TF-IDF palavra (1,2) + regressão logística: 0,2235); viés somado às log-probabilidades: c1 0,0, c234 0,0, c5 -0,15. Componentes com peso 0 (descartados pela otimização dos pesos): TF-IDF palavra (1,2) + LinearSVC, TF-IDF palavra (1,2) + ComplementNB, TF-IDF palavra (1,2) + MultinomialNB. Os parâmetros foram ajustados nas predições out-of-fold de todos os folds e estão em `modelos/melhores_modelos.json`.

**F1-macro por fold**

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | 0,4729 | 0,4679 | 0,4758 | 0,4596 | 0,4669 | **0,4686** |
| Baseline TF-IDF + regressão logística | 0,4412 | 0,4567 | 0,4590 | 0,4520 | 0,4515 | **0,4521** |

**Desempenho por classe** (predições out-of-fold de todos os folds)

| Classe | F1 | Previsto c1 | Previsto c234 | Previsto c5 |
|---|---|---|---|---|
| **c1** (real) | 0,4960 | 3270 | 1974 | 1103 |
| **c234** (real) | 0,3978 | 2222 | 2721 | 1910 |
| **c5** (real) | 0,5125 | 1346 | 2133 | 3413 |

**Ranking dos candidatos com BERT**

| # | Modelo | F1-macro (média ± dp) | Acurácia (média ± dp) |
|---|---|---|---|
| 1 | Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | 0,4686 ± 0,0056 | 0,4682 ± 0,0059 |
| 2 | Ensemble BERTimbau + SVC palavra+caractere (média ponderada + viés) | 0,4683 ± 0,0072 | 0,4677 ± 0,0070 |
| 3 | Ensemble BERTimbau + ComplementNB (média simples + viés) | 0,4676 ± 0,0079 | 0,4673 ± 0,0077 |
| 4 | Ensemble BERTimbau + SVC palavra+caractere + ComplementNB (média ponderada + viés) | 0,4670 ± 0,0062 | 0,4671 ± 0,0066 |
| 5 | Ensemble BERTimbau + SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada) | 0,4663 ± 0,0040 | 0,4706 ± 0,0036 |
| 6 | Ensemble BERTimbau + SVC palavra+caractere (média simples + viés) | 0,4663 ± 0,0065 | 0,4658 ± 0,0063 |
| 7 | Ensemble BERTimbau + SVC palavra+caractere + ComplementNB (média simples + viés) | 0,4657 ± 0,0080 | 0,4663 ± 0,0078 |
| 8 | Ensemble BERTimbau + ComplementNB (média simples) | 0,4654 ± 0,0089 | 0,4696 ± 0,0087 |
| 9 | Ensemble BERTimbau + SVC palavra+caractere + ComplementNB (média simples) | 0,4644 ± 0,0083 | 0,4686 ± 0,0085 |
| 10 | Ensemble BERTimbau + SVC palavra+caractere (média ponderada) | 0,4642 ± 0,0047 | 0,4694 ± 0,0044 |

**Cada modelo sozinho** (sem ensemble e sem ajuste de viés)

| # | Modelo | F1-macro (média ± dp) | Acurácia (média ± dp) |
|---|---|---|---|
| 1 | TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC | 0,4542 ± 0,0016 | 0,4587 ± 0,0031 |
| 2 | TF-IDF palavra (1,2) + LinearSVC | 0,4541 ± 0,0075 | 0,4572 ± 0,0081 |
| 3 | BERTimbau fine-tuning (2 épocas) | 0,4540 ± 0,0051 | 0,4601 ± 0,0029 |
| 4 | Baseline TF-IDF + regressão logística (baseline) | 0,4521 ± 0,0061 | 0,4541 ± 0,0065 |
| 5 | TF-IDF palavra (1,2) + MultinomialNB | 0,4511 ± 0,0066 | 0,4527 ± 0,0070 |
| 6 | TF-IDF palavra (1,2) + regressão logística | 0,4511 ± 0,0062 | 0,4515 ± 0,0071 |
| 7 | TF-IDF palavra (1,2) + ComplementNB | 0,4498 ± 0,0069 | 0,4510 ± 0,0076 |

O BERTimbau sozinho tem F1 0,4540 ± 0,0051, +0,2 pts em relação ao baseline. O ganho do melhor modelo vem da **combinação** do BERT com os modelos TF-IDF, que erram em textos diferentes.

## 5. Melhor modelo usando outros métodos (sem BERT, excluindo os baselines)

**Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés)**: F1-macro **0,4597 ± 0,0040**, acurácia **0,4589 ± 0,0048** (+0,8 pts sobre o baseline).

Pesos finais (TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC: 0,9422, TF-IDF palavra (1,2) + MultinomialNB: 0,0578); viés somado às log-probabilidades: c1 -0,1, c234 0,0, c5 -0,15. Componentes com peso 0 (descartados pela otimização dos pesos): TF-IDF palavra (1,2) + LinearSVC, TF-IDF palavra (1,2) + regressão logística, TF-IDF palavra (1,2) + ComplementNB.

**F1-macro por fold**

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | 0,4576 | 0,4534 | 0,4653 | 0,4603 | 0,4618 | **0,4597** |
| Baseline TF-IDF + regressão logística | 0,4412 | 0,4567 | 0,4590 | 0,4520 | 0,4515 | **0,4521** |

**Desempenho por classe**

| Classe | F1 | Previsto c1 | Previsto c234 | Previsto c5 |
|---|---|---|---|---|
| **c1** (real) | 0,4649 | 2831 | 2236 | 1280 |
| **c234** (real) | 0,4107 | 1875 | 2950 | 2028 |
| **c5** (real) | 0,5042 | 1126 | 2328 | 3438 |

**Ranking dos candidatos sem BERT**

| # | Modelo | F1-macro (média ± dp) | Acurácia (média ± dp) |
|---|---|---|---|
| 1 | Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada + viés) | 0,4597 ± 0,0040 | 0,4589 ± 0,0048 |
| 2 | Ensemble SVC palavra+caractere + ComplementNB (média ponderada + viés) | 0,4576 ± 0,0026 | 0,4571 ± 0,0034 |
| 3 | Ensemble SVC palavra+caractere + ComplementNB + MultinomialNB (média ponderada + viés) | 0,4576 ± 0,0026 | 0,4571 ± 0,0034 |
| 4 | TF-IDF palavra (1,2) + LinearSVC + ajuste de viés | 0,4573 ± 0,0073 | 0,4571 ± 0,0073 |
| 5 | Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média ponderada) | 0,4560 ± 0,0038 | 0,4596 ± 0,0049 |
| 6 | Ensemble SVC palavra + SVC palavra+caractere + LogReg + ComplementNB + MultinomialNB (média simples + viés) | 0,4560 ± 0,0050 | 0,4548 ± 0,0058 |
| 7 | TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC + ajuste de viés | 0,4559 ± 0,0042 | 0,4557 ± 0,0046 |
| 8 | Ensemble SVC palavra+caractere + ComplementNB (média ponderada) | 0,4552 ± 0,0032 | 0,4595 ± 0,0046 |
| 9 | Ensemble SVC palavra+caractere + ComplementNB + MultinomialNB (média ponderada) | 0,4552 ± 0,0032 | 0,4595 ± 0,0046 |
| 10 | Ensemble SVC palavra+caractere + ComplementNB (média simples + viés) | 0,4545 ± 0,0035 | 0,4546 ± 0,0033 |

## 6. Outros resultados do projeto (divisão única, não comparáveis com a validação cruzada)

Estes experimentos foram avaliados numa única divisão, e cada divisão tem dificuldade diferente. O mesmo baseline TF-IDF + regressão logística vai de 0,4412 no holdout agrupado a 0,4648 na divisão aleatória. Por isso não entram na seleção do melhor modelo, que exige validação cruzada. Os métodos que se destacaram aqui foram reavaliados com validação cruzada nas seções 4 e 5.

| Experimento | Avaliação | F1-macro |
|---|---|---|
| BERTimbau (treino planejado para 4 épocas), época 1 | holdout agrupado (= fold 1) | 0,4345 |
| BERTimbau (treino planejado para 4 épocas), época 2 | holdout agrupado (= fold 1) | 0,4601 |
| BERTimbau (treino planejado para 4 épocas), época 3 | holdout agrupado (= fold 1) | 0,4487 |
| BERTimbau (treino planejado para 4 épocas), época 4 | holdout agrupado (= fold 1) | 0,4474 |
| BERTimbau 2 épocas | holdout agrupado (= fold 1) | 0,4536 |
| Ensemble clássico SVC palavra+caractere + ComplementNB | divisão do baseline (80/20, semente 123) | 0,4834 |
| TF-IDF word(1,2) + SVC (tfidf_ngrams_binario.py) | split aleatório próprio (teste 4.019) | 0,4778 |
| ComplementNB (naive_bayes.py) | CV aninhado estratificado | 0,4528 |
| SelectKBest + LinearSVC (tfidf_selectkbest_linearsvc.py) | teste independente próprio | 0,4507 |
| Grid noturno: word(1,2) + LinearSVC C=0,1 | CV estratificado 5 folds | 0,4490 |
| Word2Vec CBOW 200d + LogReg (word2vec.py) | teste independente próprio | 0,4472 |

Não reavaliados com validação cruzada por falta de tempo de GPU (cerca de 11 h por configuração): label smoothing 0,1 e Legal-BERTimbau. Os scripts estão prontos (`pretreino_mlm.py`, `finetune_bertimbau.py --modelo ...`). Word2Vec ficou abaixo do TF-IDF em todas as configurações testadas.

## 7. Como funcionam os melhores modelos

### 7.1 BERTimbau (componente BERT)

1. **Modelo pré-treinado:** `neuralmind/bert-base-portuguese-cased` (BERTimbau base: 110 milhões de parâmetros, 12 camadas), pré-treinado em um grande corpus de português brasileiro. Ele já "entende" a língua antes de ver os nossos dados.
2. **Tokenização:** WordPiece, até 512 tokens. Textos longos (10,9% dos textos) mantêm os **128 primeiros e os 382 últimos tokens**, porque o fim da resposta costuma dizer se o pedido foi atendido, negado ou redirecionado.
3. **Fine-tuning:** uma camada de classificação (3 classes) é acoplada ao token `[CLS]` e o modelo inteiro é ajustado com entropia cruzada. Configuração: AdamW, lr 2e-5 com aquecimento de 10% e decaimento linear, lote efetivo 16 (8 × 2 de acumulação), precisão mista fp16, *gradient clipping* 1,0, *weight decay* 0,01.
4. **2 épocas:** com mais épocas o modelo passa a memorizar o ruído dos rótulos. No holdout, o F1 foi 0,4345 → 0,4601 → 0,4487 → 0,4474 nas épocas 1 a 4.
5. **Saída:** probabilidades para c1, c234 e c5.

**Variante com TAPT (BERTimbau-TAPT).** Antes do fine-tuning, o BERTimbau passa por um **pré-treino adaptativo à tarefa** (TAPT; Gururangan et al., 2020, *Don't Stop Pretraining*), com `pretreino_mlm.py`:

- **O que é:** continuação do pré-treino com *Masked Language Modeling*: 15% dos tokens são escondidos (80% trocados por `[MASK]`, 10% por um token aleatório e 10% mantidos) e o modelo aprende a prevê-los pelo contexto. Ex.: "Encaminhamos, na caixa [MASK], informação fornecida pela [MASK] responsável" → "Anexos", "área".
- **Sem vazamento na validação cruzada:** cada fold tem **o seu próprio TAPT**, feito só com os textos de treino daquele fold, sem rótulos. Os textos do fold de validação (e suas cópias, já que os folds são agrupados por texto) nunca entram no pré-treino. O conjunto de teste não é usado.
- **Modelo final da entrega:** depois da validação cruzada, todo o `train.xlsx` é treino. O TAPT final usa todos os textos de treino, sem rótulos, e em seguida vem o fine-tuning com 100% dos dados.
- **Por quê:** o texto do e-SIC (templates de órgãos, siglas, citações de leis) é diferente do português geral do pré-treino original. O TAPT adapta o vocabulário e o estilo antes de o modelo aprender a classificar.
- **Configuração:** 2 épocas de MLM, lr 5e-5, lote efetivo 32, até 256 tokens (64 do início + 190 do fim). Depois, o mesmo fine-tuning de 2 épocas descrito acima. Custo: ~4h20 por fold (MLM + fine-tuning) na GTX 1660 Super.

### 7.2 Modelos clássicos (componentes sem BERT)

- **TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC:** duas representações TF-IDF concatenadas. A de palavras usa unigramas e bigramas; a de caracteres usa n-gramas de 2 a 5 caracteres dentro das palavras, capturando radicais, siglas e variações como "SIC/MTb" e "Decreto nº 7.724". Ambas com `sublinear_tf`. Classificador LinearSVC (C=0,05), calibrado com `CalibratedClassifierCV` (sigmoide) para produzir probabilidades.
- **TF-IDF palavra (1,2) + ComplementNB (alpha=0,5)** e **MultinomialNB (alpha=0,1):** Naive Bayes, rápidos e fortes em "decorar" padrões frequentes de palavras.
- **TF-IDF palavra (1,2) + LinearSVC (C=0,1)** e **regressão logística (C=2)**.

### 7.3 Ensemble

Para cada texto, cada componente produz probabilidades para as 3 classes. O ensemble faz a **média ponderada** dessas probabilidades (pesos no simplex, ajustados minimizando a log-loss) e, quando indicado, soma um **viés por classe** às log-probabilidades, deslocando a fronteira de decisão para maximizar o F1-macro. A classe final é a de maior valor. O ganho vem da **diversidade**: o BERT captura sentido e contexto, enquanto o TF-IDF captura palavras e templates exatos, e cada um erra em textos diferentes.

## 8. Como reproduzir

Requisitos: Python 3.13, `pandas`, `scikit-learn`, `openpyxl`, `joblib`, `torch` (CUDA) e `transformers`. Tempos medidos numa GTX 1660 Super (6 GB).

```bash
# 1. Validação cruzada dos baselines e modelos clássicos (CPU, ~40 min)
python cv_classicos.py

# 2. Validação cruzada do BERTimbau (GPU, ~2h10 por fold)
python finetune_bertimbau.py --modo cv --epocas 2 --saida modelos/bertimbau/cv_2ep

# 2b. Validação cruzada com TAPT sem vazamento: para cada fold k, TAPT só com os textos de treino do fold
#     e fine-tuning a partir dele (GPU, ~4h20 por fold). Automatizado em modelos/bertimbau/fila_tapt_folds.sh
python pretreino_mlm.py --textos fold --fold 1 --epocas 2 --saida modelos/tapt_folds/fold_1
python finetune_bertimbau.py --modo cv --apenas-fold 1 --epocas 2 --modelo modelos/tapt_folds/fold_1 --saida modelos/bertimbau/cv_tapt_2ep
# ... repetir para os folds 2 a 5

# 3. Este relatório e a receita dos melhores modelos (modelos/melhores_modelos.json)
python gerar_relatorio.py

# 4. Modelo BERTimbau final, treinado com 100% do train.xlsx (GPU, ~2h45)
python finetune_bertimbau.py --modo final --epocas 2 --saida modelos/bertimbau/final_2ep

# 4b. Modelo final BERTimbau+TAPT: TAPT com todos os textos de treino, depois fine-tuning com 100% dos dados (GPU, ~5h30)
python pretreino_mlm.py --textos todos --epocas 2 --saida modelos/bertimbau_tapt_todos
python finetune_bertimbau.py --modo final --epocas 2 --modelo modelos/bertimbau_tapt_todos --saida modelos/bertimbau/final_tapt_2ep

# 5. Rotular o conjunto de teste com o melhor modelo (só na entrega)
python prever_ensemble.py --modelo bert --teste <arquivo_teste>.xlsx
python prever_ensemble.py --modelo outros_metodos --teste <arquivo_teste>.xlsx

# 5b. Variante usada na entrega: o mesmo ensemble com BERT, sem o ajuste de viés (maior acurácia)
python prever_ensemble.py --modelo bert --sem-vies --teste data/test1.xlsx
```

**Arquivo entregue:** `modelos/entrega/bert_sem_vies/test1.xlsx`. A nota do EP usa acurácia, e o mesmo ensemble sem o ajuste de viés tem a maior acurácia na validação cruzada (seção 1). Os pesos dos componentes são os mesmos; só o viés por classe é desligado.

O passo 5 treina os componentes clássicos em 100% dos dados, usa o BERTimbau final salvo em `modelos/bertimbau/final_2ep/final` e combina as probabilidades com os pesos e o viés de `modelos/melhores_modelos.json`.

Arquivos principais:

| Arquivo | Função |
|---|---|
| `src/divisoes.py` | Folds da validação cruzada agrupada e divisão do baseline |
| `src/modelos_classicos.py` | Definição dos modelos clássicos |
| `cv_classicos.py` | Validação cruzada dos baselines e clássicos (probabilidades out-of-fold em `modelos/cv/`) |
| `finetune_bertimbau.py` | Fine-tuning do BERTimbau (modos holdout, cv, final e prever) |
| `gerar_relatorio.py` | Comparação nos mesmos folds, ensembles com cross-fitting e este relatório |
| `prever_ensemble.py` | Rotulação do teste com o melhor modelo |
| `EXPERIMENTOS.md` | Diário completo dos experimentos e das decisões |
