#!/usr/bin/env bash
# Phase 1 on the training machine: download, list, analyse and align the human-speech corpus.
#
#     DATA=/workspace/data bash scripts/prepare_training_data.sh
#
# Every step resumes where it stopped, so after an interruption run it again.
# Needs the repository installed with its extras:
#     pip install -e ".[own,prepare,annotate]"
# Settings: DATA (required), CORPUS (default $DATA/corpus), WORKERS (default: all
# cores), DEVICE (default cuda), MIRROR (an openslr mirror, see fetch_training_data.sh).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:?set DATA to the folder for the training data}
CORPUS=${CORPUS:-$DATA/corpus}
WORKERS=${WORKERS:-$(nproc)}
DEVICE=${DEVICE:-cuda}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "1/4 downloading"
DATA=$DATA bash "$HERE/scripts/fetch_training_data.sh"

if [[ ! -f $CORPUS/report.json ]]; then
  log "2/4 listing the corpus (Hi-Fi TTS is resampled to 24 kHz here)"
  python "$HERE/scripts/build_corpus.py" --output "$CORPUS" --workers "$WORKERS" \
    --libritts "$DATA/LibriTTS_R/train-clean-100" "$DATA/LibriTTS_R/train-clean-360" --hifitts "$DATA/hi_fi_tts_v0"
else
  log "2/4 corpus already listed: $CORPUS/report.json"
fi

for split in validation train; do
  log "3/4 waveforms, spectra, pitch and energy: $split"
  tiny-voca-prepare --inputs "$CORPUS/$split.jsonl" --output "$CORPUS/features" --split "$split" \
    --config "$CORPUS/config.json" --workers "$WORKERS" --resume
done

REVISION=corpus-$(sha256sum "$CORPUS/train.jsonl" | cut -c1-12)
log "4/4 phonemes and forced alignment ($REVISION)"
tiny-voca-annotate --features "$CORPUS/features" --output "$CORPUS/annotated" --dataset-revision "$REVISION" \
  --config "$CORPUS/config.json" --device "$DEVICE" --splits train validation --resume

log "done. corpus: $CORPUS/report.json, alignment: $CORPUS/annotated"
