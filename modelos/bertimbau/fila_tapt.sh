#!/usr/bin/env bash
# Fila TAPT (opção B):
#   1. Pré-treino MLM do BERTimbau em todos os textos do train.xlsx, sem rótulos (~1h30-2h)
#   2. Validação cruzada agrupada 5 folds do BERTimbau+TAPT, 2 épocas, mesmos folds de antes (~11h)
#      -> relatório regenerado a cada fold concluído
#   3. Modelo final BERTimbau+TAPT com 100% dos dados (~2h45), para a entrega caso o TAPT vença
set -u
cd "C:/Users/diego/Downloads/PLN/PLN-grupo-4"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
B=modelos/bertimbau
TAPT=modelos/bertimbau_tapt_todos
log() { echo "[$(date '+%H:%M')] $*"; }
relatorio() { python gerar_relatorio.py > modelos/cv/log_relatorio.txt 2>&1 && log "RELATORIO atualizado" || log "ERRO relatorio"; }

log "INICIO TAPT: pre-treino MLM em todos os textos de treino"
mkdir -p $TAPT
python pretreino_mlm.py --textos todos --saida $TAPT > $TAPT/log.txt 2>&1 \
  && log "FIM TAPT -> $TAPT" || { log "ERRO TAPT (ver $TAPT/log.txt)"; exit 1; }

log "INICIO validacao cruzada BERTimbau+TAPT 2 epocas"
mkdir -p $B/cv_tapt_2ep/cv_folds
python finetune_bertimbau.py --modo cv --epocas 2 --modelo $TAPT --saida $B/cv_tapt_2ep > $B/cv_tapt_2ep/log.txt 2>&1 &
PID=$!
ULTIMO=0
while kill -0 $PID 2>/dev/null; do
  sleep 120
  ATUAL=$(ls $B/cv_tapt_2ep/cv_folds/fold_*.csv 2>/dev/null | wc -l)
  if [ "$ATUAL" != "$ULTIMO" ]; then ULTIMO=$ATUAL; log "FIM fold TAPT ($ATUAL de 5)"; relatorio; fi
done
wait $PID && log "FIM validacao cruzada TAPT" || { log "ERRO validacao cruzada TAPT (ver $B/cv_tapt_2ep/log.txt)"; exit 1; }
relatorio

log "INICIO modelo final BERTimbau+TAPT (100% dos dados)"
mkdir -p $B/final_tapt_2ep
python finetune_bertimbau.py --modo final --epocas 2 --modelo $TAPT --saida $B/final_tapt_2ep > $B/final_tapt_2ep/log.txt 2>&1 \
  && log "FIM modelo final TAPT -> $B/final_tapt_2ep/final" || log "ERRO modelo final TAPT"
relatorio
log "FILA TAPT CONCLUIDA"
