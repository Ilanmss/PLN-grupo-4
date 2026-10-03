#!/usr/bin/env bash
# Fase 2: novas variantes de transformer no MESMO holdout (espera a fila da fase 1 terminar).
#   1. TAPT: pré-treino MLM do BERTimbau nos textos de treino do holdout, depois fine-tuning
#   2. Albertina-100m PTBR (arquitetura DeBERTa)
#   3. Legal-BERTimbau-base (BERTimbau com pré-treino adicional em textos jurídicos)
# Todas com 2 épocas; label smoothing 0.1 só se ele ganhou na fase 1.
# Ao final, roda o ensemble com todos os modelos disponíveis no holdout.
set -u
cd "C:/Users/diego/Downloads/PLN/PLN-grupo-4"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
B=modelos/bertimbau
H=modelos/holdout_probabilidades
log() { echo "[$(date '+%H:%M')] $*"; }
f1_final() { python -c "import pandas as pd;print(round(pd.read_csv('$1/holdout_historico.csv')['f1_macro'].iloc[-1],4))"; }

until grep -qE "FILA CONCLUIDA|ERRO (final|holdout 2 epocas|holdout label)" $B/fila_log.txt; do sleep 60; done
if grep -q "LABEL SMOOTHING GANHOU" $B/fila_log.txt; then LS=0.1; else LS=0.0; fi
log "FASE 2 INICIADA (label smoothing = $LS)"

rodar_holdout() {  # nome, modelo
  log "INICIO holdout $1"
  mkdir -p $B/holdout_$1
  if python finetune_bertimbau.py --modo holdout --epocas 2 --label-smoothing $LS --modelo "$2" \
       --saida $B/holdout_$1 > $B/holdout_$1/log.txt 2>&1; then
    cp $B/holdout_$1/holdout_probabilidades_epoca_2.csv $H/$1.csv
    log "FIM holdout $1 F1=$(f1_final $B/holdout_$1)"
  else
    log "ERRO holdout $1 (ver $B/holdout_$1/log.txt)"
  fi
}

# 1. TAPT
log "INICIO pre-treino MLM (TAPT)"
mkdir -p modelos/bertimbau_tapt
if python pretreino_mlm.py --saida modelos/bertimbau_tapt > modelos/bertimbau_tapt/log.txt 2>&1; then
  log "FIM pre-treino MLM"
  rodar_holdout bertimbau_tapt modelos/bertimbau_tapt
else
  log "ERRO pre-treino MLM (ver modelos/bertimbau_tapt/log.txt)"
fi

# 2 e 3. Outros modelos pré-treinados em português
rodar_holdout albertina100m PORTULAN/albertina-100m-portuguese-ptbr-encoder
rodar_holdout legal_bertimbau rufimelo/Legal-BERTimbau-base

# 4. Ensemble com tudo que existe no holdout (BERTimbau da fase 1 + variantes + clássicos)
[ -f $B/holdout_2ep/holdout_probabilidades_epoca_2.csv ] && cp $B/holdout_2ep/holdout_probabilidades_epoca_2.csv $H/bertimbau.csv
[ -f $B/holdout_2ep_ls01/holdout_probabilidades_epoca_2.csv ] && cp $B/holdout_2ep_ls01/holdout_probabilidades_epoca_2.csv $H/bertimbau_ls01.csv
MODELOS=""
for f in $H/*.csv; do n=$(basename $f .csv); [ "$n" = resumo_classicos ] || MODELOS="$MODELOS $n=$f"; done
log "INICIO ensemble com:$MODELOS"
mkdir -p modelos/ensemble/fase2_todos
python ensemble_holdout.py --modelos $MODELOS --saida modelos/ensemble/fase2_todos > modelos/ensemble/fase2_todos/log.txt 2>&1 \
  && log "FIM ensemble -> modelos/ensemble/fase2_todos/relatorio.txt" || log "ERRO ensemble"
log "FASE 2 CONCLUIDA"
