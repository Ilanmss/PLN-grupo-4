#!/usr/bin/env bash
# TAPT SEM VAZAMENTO: um pré-treino MLM por fold, só com os textos de treino daquele fold.
#   Para cada fold k (1..5):
#     1. pretreino_mlm.py --textos fold --fold k   (MLM 2 épocas nos textos de treino do fold k, sem rótulos; ~2h10)
#     2. finetune_bertimbau.py --modo cv --apenas-fold k a partir desse modelo (2 épocas; ~2h10)
#     3. apaga o modelo TAPT do fold (guarda o histórico) e regenera o relatório
#   Depois da validação cruzada (todo o train.xlsx vira treino), modelo final da entrega:
#     4. pretreino_mlm.py --textos todos (MLM 2 épocas em todos os textos de treino, sem rótulos; ~2h40)
#     5. finetune_bertimbau.py --modo final (2 épocas; ~2h45)
# O conjunto de teste nunca é usado. Execução retomável: folds já concluídos são pulados.
set -u
cd "C:/Users/diego/Downloads/PLN/PLN-grupo-4"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
B=modelos/bertimbau
T=modelos/tapt_folds
EPOCAS_MLM=2
log() { echo "[$(date '+%H:%M')] $*"; }
relatorio() { python gerar_relatorio.py > modelos/cv/log_relatorio.txt 2>&1 && log "RELATORIO atualizado" || log "ERRO relatorio"; }

mkdir -p $T $B/cv_tapt_2ep/cv_folds
for k in 1 2 3 4 5; do
  if [ -f $B/cv_tapt_2ep/cv_folds/fold_$k.csv ]; then log "fold $k do TAPT ja concluido, pulando"; continue; fi
  log "INICIO TAPT fold $k (MLM so com textos de treino do fold)"
  python pretreino_mlm.py --textos fold --fold $k --epocas $EPOCAS_MLM --saida $T/fold_$k > $T/log_mlm_fold_$k.txt 2>&1 \
    || { log "ERRO TAPT fold $k (ver $T/log_mlm_fold_$k.txt)"; exit 1; }
  cp $T/fold_$k/historico_mlm.json $T/historico_mlm_fold_$k.json
  log "INICIO fine-tuning fold $k a partir do TAPT do fold"
  python finetune_bertimbau.py --modo cv --apenas-fold $k --epocas 2 --modelo $T/fold_$k --saida $B/cv_tapt_2ep \
    > $T/log_finetune_fold_$k.txt 2>&1 || { log "ERRO fine-tuning fold $k (ver $T/log_finetune_fold_$k.txt)"; exit 1; }
  rm -rf $T/fold_$k
  log "FIM fold $k do TAPT"
  relatorio
done
log "FIM validacao cruzada TAPT sem vazamento"

log "INICIO TAPT final (todos os textos de treino, sem rotulos)"
mkdir -p modelos/bertimbau_tapt_todos
python pretreino_mlm.py --textos todos --epocas $EPOCAS_MLM --saida modelos/bertimbau_tapt_todos > modelos/bertimbau_tapt_todos/log.txt 2>&1 \
  || { log "ERRO TAPT final"; exit 1; }
log "INICIO modelo final BERTimbau+TAPT (100% dos dados)"
mkdir -p $B/final_tapt_2ep
python finetune_bertimbau.py --modo final --epocas 2 --modelo modelos/bertimbau_tapt_todos --saida $B/final_tapt_2ep \
  > $B/final_tapt_2ep/log.txt 2>&1 && log "FIM modelo final TAPT -> $B/final_tapt_2ep/final" || log "ERRO modelo final TAPT"
relatorio
log "FILA TAPT SEM VAZAMENTO CONCLUIDA"
