#!/usr/bin/env bash
# Fila da noite: validação cruzada agrupada (5 folds) para o relatório de resultados.
#   - GPU: BERTimbau 2 épocas, folds 2-5 (o fold 1 é o holdout de 2 épocas já executado, mesma divisão e configuração)
#   - CPU (em paralelo): validação cruzada dos baselines e dos modelos clássicos (cv_classicos.py)
#   - A cada fold concluído, o relatório RELATORIO_RESULTADOS.md é regenerado (gerar_relatorio.py)
set -u
cd "C:/Users/diego/Downloads/PLN/PLN-grupo-4"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
B=modelos/bertimbau
log() { echo "[$(date '+%H:%M')] $*"; }

log "INICIO validacao cruzada BERTimbau 2 epocas (folds 2-5)"
python finetune_bertimbau.py --modo cv --epocas 2 --saida $B/cv_2ep > $B/cv_2ep/log.txt 2>&1 &
PID_BERT=$!

log "INICIO validacao cruzada classicos e baselines (CPU)"
python cv_classicos.py > modelos/cv/log_classicos.txt 2>&1 && log "FIM validacao cruzada classicos" || log "ERRO validacao cruzada classicos"
python gerar_relatorio.py > modelos/cv/log_relatorio.txt 2>&1 && log "RELATORIO atualizado" || log "ERRO relatorio"

# Regenera o relatório sempre que um novo fold do BERT termina
ULTIMO=$(ls $B/cv_2ep/cv_folds/fold_*.csv 2>/dev/null | wc -l)
while kill -0 $PID_BERT 2>/dev/null; do
  sleep 120
  ATUAL=$(ls $B/cv_2ep/cv_folds/fold_*.csv 2>/dev/null | wc -l)
  if [ "$ATUAL" != "$ULTIMO" ]; then
    ULTIMO=$ATUAL
    log "FIM fold BERT ($ATUAL de 5 folds prontos)"
    python gerar_relatorio.py > modelos/cv/log_relatorio.txt 2>&1 && log "RELATORIO atualizado" || log "ERRO relatorio"
  fi
done
wait $PID_BERT && log "FIM validacao cruzada BERTimbau" || log "ERRO validacao cruzada BERTimbau (ver $B/cv_2ep/log.txt)"
python gerar_relatorio.py > modelos/cv/log_relatorio.txt 2>&1 && log "RELATORIO FINAL gerado" || log "ERRO relatorio"
log "FILA DA NOITE CONCLUIDA"
