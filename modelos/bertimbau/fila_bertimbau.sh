#!/usr/bin/env bash
# Fila de experimentos BERTimbau (roda após o holdout de 4 épocas terminar)
set -u
cd "C:/Users/diego/Downloads/PLN/PLN-grupo-4"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
B=modelos/bertimbau
log() { echo "[$(date '+%H:%M')] $*"; }


f1_final() { python -c "import pandas as pd;print(round(pd.read_csv('$1/holdout_historico.csv')['f1_macro'].iloc[-1],4))"; }

# 1. Modelo final, 2 épocas, 100% dos dados (sem rotular o teste)
log "INICIO final 2 epocas"
mkdir -p $B/final_2ep
python finetune_bertimbau.py --modo final --epocas 2 --saida $B/final_2ep > $B/final_2ep/log.txt 2>&1 \
  && log "FIM final 2 epocas -> $B/final_2ep/final" || { log "ERRO final 2 epocas"; exit 1; }

# 2. Holdout 2 épocas sem label smoothing (referência justa: o lr decai até zero em 2 épocas)
log "INICIO holdout 2 epocas"
mkdir -p $B/holdout_2ep
python finetune_bertimbau.py --modo holdout --epocas 2 --saida $B/holdout_2ep > $B/holdout_2ep/log.txt 2>&1 \
  || { log "ERRO holdout 2 epocas"; exit 1; }
F1_BASE=$(f1_final $B/holdout_2ep)
log "FIM holdout 2 epocas F1=$F1_BASE"

# 3. Ensemble com LinearSVC (CPU, em paralelo com o próximo treino na GPU)
( python ensemble_bertimbau_svc.py --bert $B/holdout_2ep/holdout_probabilidades_epoca_2.csv \
    --saida modelos/ensemble_bertimbau_svc/holdout_2ep > modelos/ensemble_bertimbau_svc_2ep_log.txt 2>&1 \
  && log "FIM ensemble (2ep) -> modelos/ensemble_bertimbau_svc/holdout_2ep/relatorio.txt" \
  || log "ERRO ensemble (2ep)" ) &

# 4. Holdout 2 épocas com label smoothing 0.1
log "INICIO holdout 2 epocas label smoothing 0.1"
mkdir -p $B/holdout_2ep_ls01
python finetune_bertimbau.py --modo holdout --epocas 2 --label-smoothing 0.1 --saida $B/holdout_2ep_ls01 \
  > $B/holdout_2ep_ls01/log.txt 2>&1 || { log "ERRO holdout label smoothing"; wait; exit 1; }
F1_LS=$(f1_final $B/holdout_2ep_ls01)
log "FIM holdout label smoothing F1=$F1_LS (sem LS: $F1_BASE)"
wait

# 5. Se o label smoothing ganhou: ensemble com ele e retreino do modelo final
if python -c "import sys;sys.exit(0 if $F1_LS > $F1_BASE else 1)"; then
  log "LABEL SMOOTHING GANHOU; rodando ensemble e final com LS"
  python ensemble_bertimbau_svc.py --bert $B/holdout_2ep_ls01/holdout_probabilidades_epoca_2.csv \
    --saida modelos/ensemble_bertimbau_svc/holdout_2ep_ls01 > modelos/ensemble_bertimbau_svc_ls01_log.txt 2>&1 &
  mkdir -p $B/final_2ep_ls01
  python finetune_bertimbau.py --modo final --epocas 2 --label-smoothing 0.1 --saida $B/final_2ep_ls01 \
    > $B/final_2ep_ls01/log.txt 2>&1 && log "FIM final LS -> $B/final_2ep_ls01/final" || log "ERRO final LS"
  wait
else
  log "LABEL SMOOTHING NAO GANHOU; modelo final continua $B/final_2ep/final"
fi
log "FILA CONCLUIDA"
