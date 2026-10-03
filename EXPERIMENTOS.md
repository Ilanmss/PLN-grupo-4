# Diário de experimentos — melhoria do F1

Documento vivo com o diagnóstico, os experimentos e as decisões para superar o baseline oficial
(TF-IDF + regressão logística). Atualizado em 29/09/2026.

## 1. Diagnóstico do problema

| Achado | Evidência | Consequência |
|---|---|---|
| **Rótulos muito ruidosos** | 291 textos idênticos têm rótulos diferentes (1.682 linhas); 90 aparecem nas 3 classes. Prever sempre a classe majoritária desses textos acerta só 53%. | Existe um teto de desempenho baixo; F1 ~0,50 já seria muito bom. |
| **Parte do conteúdo não está no texto** | "Resposta em anexo." e "Prezado (a) Sr. (a)," aparecem dezenas de vezes nas 3 classes. | Esses casos são imprevisíveis por qualquer modelo. |
| **c234 é a classe mais difícil** | F1 de c234 fica em ~0,38 em todos os modelos, contra ~0,45–0,53 das outras. | Junta as notas 2, 3 e 4 e é ambígua por natureza. |
| **Vazamento na validação** | `StratifiedKFold` dá F1 0,4595; `GroupKFold` por texto dá 0,4544 no mesmo modelo. | Cópias do mesmo template no treino e na validação inflam o resultado. Todos os novos experimentos usam **GroupKFold agrupado por texto**. |
| **Grid de TF-IDF saturado** | No grid noturno, o 1º e o 15º lugar diferem em 0,0016, com desvio entre folds de ~0,007. | Mais grid search em TF-IDF não traz ganho real. |

Análises: `analise_duplicatas/` (duplicatas conflitantes, efeito da deduplicação).

**Deduplicação** (`criar_train_dedup.py`): testada e **descartada**, porque não melhorou nenhum modelo (diferenças de −0,0002 a −0,0044).

## 2. Protocolo de avaliação comum

Os resultados antigos do projeto usaram divisões diferentes (aleatórias, sementes 123 e 42, com vazamento
de duplicatas), então **não são comparáveis entre si**. A partir de agora, todos os modelos são avaliados no
**mesmo holdout**:

- `GroupKFold(n_splits=5, shuffle=True, random_state=42)`, agrupado por texto normalizado (`src/deduplicacao.chave_texto`), fold 1;
- 16.116 textos de treino e 3.976 de validação; nenhum texto aparece nos dois lados;
- o conjunto de teste não rotulado **não é usado** em nenhuma etapa de desenvolvimento.

## 3. Ranking no holdout comum

| # | Modelo | F1-macro | Acurácia | Script |
|---|---|---|---|---|
| 1 | **Ensemble: BERTimbau (época 4) + SVC palavra+caractere + ComplementNB** (média ponderada) | **0,4637** | **0,4685** | `ensemble_holdout.py` |
| 2 | BERTimbau fine-tuning, época 2 (de um treino planejado para 4 épocas) | 0,4601 | 0,4588 | `finetune_bertimbau.py` |
| 3 | TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC calibrado | 0,4552 | 0,4610 | `classicos_holdout.py` |
| 4 | TF-IDF palavra (1,2) + ComplementNB | 0,4496 | 0,4522 | `classicos_holdout.py` |
| 5 | BERTimbau, época 4 (sobreajustado) | 0,4474 | 0,4502 | `finetune_bertimbau.py` |
| 6 | TF-IDF palavra (1,2) + LinearSVC calibrado (melhor do grid noturno) | 0,4472 | 0,4515 | `classicos_holdout.py` |
| 6 | TF-IDF palavra (1,2) + regressão logística | 0,4472 | 0,4477 | `classicos_holdout.py` |
| 8 | **Baseline oficial**: TF-IDF padrão + regressão logística | 0,4412 | 0,4429 | — |

Os ensembles são medidos de forma honesta: pesos escolhidos numa metade do holdout e medidos na outra,
com 5 repetições. **Ganho atual sobre o baseline oficial: +2,3 pontos de F1 e +2,6 de acurácia.**

### Resultados antigos do projeto (divisões diferentes; não comparáveis com a tabela acima)

| Experimento | Avaliação | F1-macro |
|---|---|---|
| Baselines (`baselines.py`) | split aleatório 80/20, semente 123 | LR 0,4648 · BoW 0,4494 · classe majoritária 0,1703 |
| TF-IDF binário / SVC word12 (`tfidf_ngrams_binario.py`) | teste de 4.019 linhas, split aleatório | SVC 0,4778 · LR 0,4477–0,4583 |
| Grid noturno, 200 configurações | CV estratificado 5 folds | melhor 0,4490 (LinearSVC word12, C=0,1) |
| Naive Bayes (`naive_bayes.py`) | CV aninhado · teste | ComplementNB 0,4528 · 0,4473 |
| SelectKBest + LinearSVC | teste independente | 0,4507 |
| Word2Vec + LR (`word2vec.py`) | teste independente | melhor 0,4472 (sempre abaixo do TF-IDF) |

A leitura correta é a diferença para o baseline **na mesma divisão**: no split antigo o SVC ficou
+1,3 ponto acima do baseline; no holdout comum, o ensemble atual fica +2,3 pontos acima.

### Conferência na divisão do baseline oficial (`baselines.py`: aleatória 80/20, estratificada, semente 123)

| Modelo | F1-macro | Acurácia |
|---|---|---|
| Baseline TF-IDF + LogReg (reproduzido; bate com `modelos/baselines/resultados_baselines.txt`) | 0,4648 | 0,4675 |
| TF-IDF palavra+caractere + LinearSVC | 0,4696 | 0,4740 |
| TF-IDF + ComplementNB | 0,4762 | 0,4770 |
| **Ensemble SVC + ComplementNB** (média simples, sem BERT) | **0,4834** | **0,4854** |

Nas duas divisões o ganho sobre o baseline é de ~2 pontos (+1,9 aqui; +2,3 no holdout agrupado).
Os valores absolutos são mais altos nesta divisão porque ela tem vazamento de duplicatas entre treino e teste.

## 4. BERTimbau: curva de épocas

`neuralmind/bert-base-portuguese-cased`, lr 2e-5, lote efetivo 16, fp16, truncamento início (128) + fim (382).

| Época | Perda de treino | F1 (validação) | Acurácia |
|---|---|---|---|
| 1 | 1,071 | 0,4345 | 0,4452 |
| **2** | 1,004 | **0,4601** | **0,4588** |
| 3 | 0,892 | 0,4487 | 0,4532 |
| 4 | 0,768 | 0,4474 | 0,4502 |

A partir da época 3 o modelo **memoriza o ruído dos rótulos**: a perda de treino cai, mas o F1 de validação
também cai. Decisão: **2 épocas**. Cada época leva ~65 min no holdout e ~82 min com 100% dos dados (GTX 1660 Super).

## 5. Plano de melhoria em execução

Escolhido por ganho esperado e custo de GPU (~2h por experimento):

| Etapa | Hipótese | Status |
|---|---|---|
| Modelo final BERTimbau, 2 épocas, 100% dos dados | Garante uma entrega acima do baseline | ✅ `modelos/bertimbau/final_2ep/final` |
| Holdout de 2 épocas | O lr decai até zero em 2 épocas, então o modelo é diferente da época 2 de um treino de 4 | ⏳ em execução |
| Label smoothing 0,1 | Reduz a memorização de rótulos ruidosos | ⏳ na fila |
| Ensemble com clássicos | Transformer e TF-IDF erram em casos diferentes | ✅ código pronto; +0,4 a +1 ponto |
| **TAPT**: pré-treino MLM nos textos do e-SIC (`pretreino_mlm.py`) | Adaptar o BERTimbau ao jargão administrativo costuma render +1 a +2 pontos (Gururangan et al., 2020) | ⏳ fase 2 |
| **Albertina-100m PTBR** (DeBERTa) | Arquitetura mais recente; diversidade para o ensemble | ⏳ fase 2 |
| **Legal-BERTimbau** | Pré-treino em textos jurídicos, próximos ao domínio | ⏳ fase 2 |
| Ensemble de todos os modelos | Combinação de modelos diversos | ⏳ fase 2, automático |
| Modelos finais dos componentes vencedores e rotulação do teste | Entrega | depois da fase 2 |

### Descartado, com justificativa

- **Deduplicação do treino**: testada, sem ganho (seção 1).
- **BERTimbau-large**: não cabe na GPU de 6 GB para treino (pesos + gradientes + Adam ≈ 5,4 GB antes das ativações).
- **Ajuste de viés por classe**: testado no ensemble, piora ou não muda o F1 (−0,003 a +0,002).
- **Stacking com regressão logística**: pior que a média ponderada (0,4567 contra 0,4637).
- **Mais grid search em TF-IDF**: saturado (seção 1).

## 5.1 Atualização 29/09, 23h40: protocolo passa a ser a validação cruzada

A pedido do grupo, os resultados do relatório passam a usar **validação cruzada agrupada (5 folds)**,
com os mesmos folds para todos os modelos (`src/divisoes.folds_cv`). O fold 1 é idêntico ao holdout usado até aqui.

- **BERTimbau 2 épocas, holdout (= fold 1):** F1 0,4536. Um treino planejado para 4 épocas deu 0,4601 na época 2, o que mostra variação entre execuções de ~±0,006.
- **Ensemble BERT + SVC palavra+caractere + ComplementNB no holdout:** média ponderada 0,4671; com ajuste de viés, 0,4745.
- **Ensemble clássico salvo** (`ensemble_classico.py`, `modelos/ensemble_classico/`): F1 0,4834 na divisão do baseline, mas 0,4520 no holdout agrupado. O Naive Bayes se beneficia de templates repetidos entre treino e teste.
- **Interrompidos** para priorizar a validação cruzada do BERT (~2h10 por fold): label smoothing 0,1 (parado na época 1), TAPT, Albertina e Legal-BERTimbau. Scripts prontos para quando houver tempo de GPU.
- **Fila da noite** (`modelos/bertimbau/fila_cv_noite.sh`): validação cruzada do BERTimbau nos folds 2 a 5 (GPU) e dos baselines e clássicos (`cv_classicos.py`, CPU). O `gerar_relatorio.py` regenera **`RELATORIO_RESULTADOS.md`** a cada fold concluído.

## 5.2 Resultado da validação cruzada (30/09, 08h46) e TAPT

Resultados completos em **`RELATORIO_RESULTADOS.md`** (validação cruzada agrupada, 5 folds):

| Modelo | F1-macro | Acurácia |
|---|---|---|
| Baseline TF-IDF + regressão logística | 0,4521 ± 0,0061 | 0,4541 |
| BERTimbau sozinho | 0,4540 ± 0,0051 | 0,4601 |
| Melhor sem BERT (ensemble de clássicos, média ponderada + viés) | 0,4597 ± 0,0040 | 0,4589 |
| **Melhor com BERT** (BERTimbau + SVC palavra+caractere + LogReg, média ponderada + viés) | **0,4686 ± 0,0056** | 0,4682 |

Conclusões: sozinho, o BERT empata com o baseline, e o ganho vem do ensemble. O holdout antigo (fold 1) era o fold
mais difícil para o baseline, o que inflava os ganhos medidos antes.

**TAPT.** A primeira versão (opção B, com um único pré-treino em todos os textos de treino) foi **interrompida às
11h** a pedido do grupo, porque os textos dos folds de validação entravam no pré-treino. Versão atual, **sem vazamento**
(iniciada em 30/09, 11h19; `modelos/bertimbau/fila_tapt_folds.sh`):
1. Para cada fold k: `pretreino_mlm.py --textos fold --fold k`, com MLM de 2 épocas **só nos textos de treino do fold k**,
   sem rótulos. Depois, `finetune_bertimbau.py --modo cv --apenas-fold k` a partir desse modelo. São ~4h20 por fold.
2. Modelo final da entrega: TAPT em todos os textos de treino (legítimo depois da CV, quando todo o `train.xlsx` é
   treino) e fine-tuning com 100% dos dados.
3. O MLM foi reduzido de 3 para 2 épocas para caber no prazo: na primeira época, a perda MLM já caiu de 3,13 para ~1,5.
4. O conjunto de teste nunca é usado.

**Resultado do fold 1 e decisão (30/09, 15h40): TAPT interrompido.** Critérios definidos antes de ver o resultado:
ganho ≥ +0,006 continua; empate (±0,006) é decidido pelo ensemble e pela diversidade; perda ≤ −0,006 para.

| Fold 1 (médias simples, sem ajuste) | F1-macro | Acurácia |
|---|---|---|
| Baseline TF-IDF + LogReg | 0,4412 | 0,4429 |
| BERTimbau | 0,4536 | 0,4590 |
| BERTimbau-TAPT | 0,4546 | 0,4603 |
| BERT + TAPT | 0,4573 | 0,4633 |
| BERT + SVC palavra+caractere + LogReg | 0,4691 | 0,4721 |
| TAPT + SVC palavra+caractere + LogReg | 0,4672 | 0,4701 |
| BERT + TAPT + SVC palavra+caractere + LogReg | 0,4690 | 0,4728 |

- A adaptação ao domínio funcionou: a perda MLM nos textos de validação do fold (fora do pré-treino) caiu de 3,07 para 0,80 (época 1) e 0,75 (época 2).
- Mas, na classificação, o TAPT empata com o BERT (+0,001) e **não soma no ensemble** (−0,0001).
- O motivo é a falta de diversidade: o BERT e o TAPT concordam em 83,5% das predições (correlação das probabilidades 0,937), contra ~69% de concordância com o SVC.
- Conclusão: entender melhor o texto não resolve o principal limitador, que é o ruído dos rótulos. Continuar custaria ~22h de GPU para um ganho esperado próximo de zero.
- Retomável com `bash modelos/bertimbau/fila_tapt_folds.sh` (continua do fold 2).

**Albertina-100m PTBR (DeBERTa), checagem de 1 fold (30/09, 17h50–23h11): não continua.** Mesmos critérios do TAPT.

Ajustes técnicos necessários (em `finetune_bertimbau.py`):
- O checkpoint vem em **bfloat16**, e o `transformers` atual carrega no tipo salvo. Os pesos passaram a ser convertidos para float32 (`.float()`).
- **DeBERTa + fp16:** a atenção preenchia o padding com o mínimo de float32, que estoura em float16. A função `corrigir_deberta_fp16()` troca só essa linha, sem alterar a biblioteca. Diferença para a execução em float32: 0,00015 nos logits.
- O lote 8 × 512 tokens precisava de 8,3 GB. Usei lote 4 × acumulação 4 (pico de 5,7 GB), com o mesmo lote efetivo de 16.
- Custo: 2h15–2h40 por época (o dobro do BERTimbau). O tokenizador fragmenta mais: mediana de 272 tokens contra 182, com 24% dos textos truncados contra 11%.

| Fold 1 (médias simples) | F1-macro | Acurácia |
|---|---|---|
| Baseline TF-IDF + LogReg | 0,4412 | 0,4429 |
| BERTimbau | 0,4536 | 0,4590 |
| Albertina | 0,4370 | 0,4497 |
| BERT + Albertina | 0,4555 | 0,4645 |
| BERT + SVC palavra+caractere + LogReg | 0,4691 | 0,4721 |
| BERT + Albertina + SVC palavra+caractere + LogReg | 0,4662 | 0,4708 |
| BERT + 0,25 × Albertina + SVC palavra+caractere + LogReg | 0,4692 | 0,4731 |

- **Sozinho, é pior:** −0,017 contra o BERTimbau, abaixo até do baseline neste fold.
- **É mais diverso que o TAPT** (concorda com o BERT em 74%, contra 83,5%; correlação 0,837), mas fraco demais para somar: com peso igual, piora o ensemble (−0,003); com peso 0,25, empata (+0,0001).
- Custo evitado: ~22h de GPU (folds 2 a 5 e o modelo final).
- **Conclusão geral das variantes de transformer:** nem a adaptação ao domínio (TAPT) nem outra arquitetura (DeBERTa) superaram o BERTimbau. O ganho do projeto vem da combinação do BERTimbau com modelos TF-IDF.

O `gerar_relatorio.py` passou a considerar o BERTimbau-TAPT (sozinho e em ensembles). Enquanto a CV do TAPT não
fecha os 5 folds, o relatório mostra só o resultado parcial, comparado ao BERTimbau nos mesmos folds.

## 5.3 Entrega: rotulação do conjunto de teste (03/10)

**Arquivo de entrega:** `modelos/entrega/bert_sem_vies/test1.xlsx`, gerado por
`python prever_ensemble.py --modelo bert --sem-vies --teste data/test1.xlsx`.

O grupo escolheu a variante **sem ajuste de viés**, porque a nota do EP usa acurácia e essa variante tem a maior
acurácia na validação cruzada (0,4706 contra 0,4682). Os pesos são os mesmos da receita `bert`; só o viés é desligado.
A versão com viés (maior F1-macro) foi gerada antes e continua em `modelos/entrega/bert/test1.xlsx`, **mas não é a da entrega**.

- **Modelo:** receita `bert` de `modelos/melhores_modelos.json`: BERTimbau (peso 0,4787) + SVC palavra+caractere (0,2978) + regressão logística (0,2235), média ponderada. O viés da receita (−0,15 em c5) não é aplicado na entrega.
- **Treino com 100% do `train.xlsx` (20.092 textos):** o BERTimbau final já estava salvo em `modelos/bertimbau/final_2ep/final`; os dois clássicos foram treinados na hora e guardados em `modelos/finais/`.
- **Arquivo:** 900 linhas, mesma aba (`test1`), mesmas colunas e mesma ordem do original; só `clarity` preenchida. Conferido célula a célula contra o original.
- **Predições da entrega (sem viés):** c1 270, c234 276, c5 354. 88 textos do teste são idênticos a textos do treino.
- **Diferença para a versão com viés** (c1 282, c234 323, c5 295): 59 dos 900 rótulos, todos passando para c5 (47 vindos de c234 e 12 de c1).
- **Comparação na validação cruzada:** sem viés a acurácia é 0,4706 contra 0,4682 (melhor em 3 de 5 folds) e o F1 é 0,4663 contra 0,4686 (pior em 4 de 5 folds). A diferença fica dentro do ruído entre folds.
- As probabilidades das duas execuções são idênticas, e os rótulos da entrega correspondem ao argmax da média ponderada.

## 6. Como reproduzir

```bash
python classicos_holdout.py                                        # probabilidades dos clássicos no holdout
python finetune_bertimbau.py --modo holdout --epocas 2             # BERTimbau no holdout
python pretreino_mlm.py --saida modelos/bertimbau_tapt             # TAPT
python finetune_bertimbau.py --modo holdout --epocas 2 --modelo modelos/bertimbau_tapt
python ensemble_holdout.py --modelos nome=arquivo.csv ... --saida modelos/ensemble/<nome>
python finetune_bertimbau.py --modo final --epocas 2               # modelo final (100% dos dados)
```

As filas de GPU estão em `modelos/bertimbau/fila_bertimbau.sh` (fase 1) e `modelos/bertimbau/fila_fase2.sh`
(fase 2). O andamento fica em `modelos/bertimbau/fila_log.txt`.
