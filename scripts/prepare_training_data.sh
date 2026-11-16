#!/usr/bin/env bash
# Phase 1 on the training machine: download, list, analyse and align the human-speech corpus.
#
#     DATA=/workspace/data bash scripts/prepare_training_data.sh
#
# Every step resumes where it stopped, so after an interruption run it again.
# Needs the repository installed with its extras:
#     pip install -e ".[own,prepare,annotate]" git+https://github.com/sarulab-speech/UTMOSv2
# Settings: DATA (required), CORPUS (default $DATA/corpus), WORKERS (default: all
# cores), DEVICE (default cuda), MIRROR (an openslr mirror, see fetch_training_data.sh),
# QUALITY_CLIPS (clips scored per speaker, default 10), QUALITY_DROP (fraction of
# speakers left out for sound quality, default 0.2). To change the quality cut,
# delete $CORPUS/report.json and $CORPUS/features before running again.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:?set DATA to the folder for the training data}
CORPUS=${CORPUS:-$DATA/corpus}
WORKERS=${WORKERS:-$(nproc)}
DEVICE=${DEVICE:-cuda}
QUALITY_CLIPS=${QUALITY_CLIPS:-10}
QUALITY_DROP=${QUALITY_DROP:-0.2}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "1/5 downloading"
DATA=$DATA bash "$HERE/scripts/fetch_training_data.sh"

SOURCES=(--libritts "$DATA/LibriTTS_R/train-clean-100" "$DATA/LibriTTS_R/train-clean-360" --hifitts "$DATA/hi_fi_tts_v0"
         --audio-dir "$DATA/hifi-24k" --workers "$WORKERS")
if [[ ! -f $CORPUS/all/report.json ]]; then
  log "2/5 listing every usable clip (Hi-Fi TTS is resampled to 24 kHz here)"
  python "$HERE/scripts/build_corpus.py" --output "$CORPUS/all" "${SOURCES[@]}"
fi

log "3/5 scoring $QUALITY_CLIPS clips per speaker and leaving out the worst $QUALITY_DROP of speakers"
python "$HERE/scripts/rank_speakers.py" --corpus "$CORPUS/all" --clips "$QUALITY_CLIPS" --drop-fraction "$QUALITY_DROP" \
  --device "$DEVICE"
if [[ ! -f $CORPUS/report.json ]]; then
  python "$HERE/scripts/build_corpus.py" --output "$CORPUS" "${SOURCES[@]}" --exclude-speakers "$CORPUS/all/excluded-speakers.txt"
fi

for split in validation train; do
  log "4/5 waveforms, spectra, pitch and energy: $split"
  tiny-voca-prepare --inputs "$CORPUS/$split.jsonl" --output "$CORPUS/features" --split "$split" \
    --config "$CORPUS/config.json" --workers "$WORKERS" --resume
done

REVISION=corpus-$(sha256sum "$CORPUS/train.jsonl" | cut -c1-12)
log "5/5 phonemes and forced alignment ($REVISION)"
tiny-voca-annotate --features "$CORPUS/features" --output "$CORPUS/annotated" --dataset-revision "$REVISION" \
  --config "$CORPUS/config.json" --device "$DEVICE" --splits train validation --resume

log "done. corpus: $CORPUS/report.json, alignment: $CORPUS/annotated"
