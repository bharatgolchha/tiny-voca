#!/usr/bin/env bash
# Phase 1 on the training machine: download, list, analyse and align the human-speech corpus.
#
#     DATA=/workspace/data bash scripts/prepare_training_data.sh
#
# Every step resumes where it stopped, so after an interruption run it again.
# Needs the repository installed with its extras:
#     pip install -e ".[own,prepare,annotate]" git+https://github.com/sarulab-speech/UTMOSv2
# Step 5 also needs DIZIONE_DATA (see below). Set HF_HOME to a folder on the persistent
# volume (for example /workspace/hf) so downloaded models survive between machines.
# Settings: DATA (required), CORPUS (default $DATA/corpus), WORKERS (default: all
# cores), DEVICE (default cuda), MIRROR (an openslr mirror, see fetch_training_data.sh),
# QUALITY_CLIPS (clips scored per speaker, default 10), QUALITY_DROP (fraction of
# speakers left out for sound quality, default 0.2). To change the quality cut,
# delete $CORPUS/report.json and $CORPUS/features before running again.
#
# STEPS picks which steps run (default "1 2 3 4 5"). Steps 1, 2 and 4 need only
# processor cores; 3 (speaker scoring) and 5 (alignment) want a GPU. To pay for a
# GPU only when it is used, on a shared volume run STEPS="1 2" on a CPU machine,
# STEPS=3 on the GPU, STEPS=4 on the CPU machine and STEPS=5 on the GPU.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:?set DATA to the folder for the training data}
CORPUS=${CORPUS:-$DATA/corpus}
# nproc can report the host's cores inside a container; the cgroup quota is the real limit.
WORKERS=${WORKERS:-$(python -m tiny_voca.cpus 2>/dev/null || nproc)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$WORKERS} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$WORKERS}
DEVICE=${DEVICE:-cuda}
QUALITY_CLIPS=${QUALITY_CLIPS:-10}
QUALITY_DROP=${QUALITY_DROP:-0.2}
STEPS=${STEPS:-1 2 3 4 5}
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }

if step 1; then
  log "1/5 downloading"
  DATA=$DATA bash "$HERE/scripts/fetch_training_data.sh"
fi

SOURCES=(--libritts "$DATA/LibriTTS_R/train-clean-100" "$DATA/LibriTTS_R/train-clean-360" --hifitts "$DATA/hi_fi_tts_v0"
         --audio-dir "$DATA/hifi-24k" --workers "$WORKERS")
if step 2 && [[ ! -f $CORPUS/all/report.json ]]; then
  log "2/5 listing every usable clip (Hi-Fi TTS is resampled to 24 kHz here)"
  python "$HERE/scripts/build_corpus.py" --output "$CORPUS/all" "${SOURCES[@]}"
fi

if step 3; then
  log "3/5 scoring $QUALITY_CLIPS clips per speaker and leaving out the worst $QUALITY_DROP of speakers"
  python "$HERE/scripts/rank_speakers.py" --corpus "$CORPUS/all" --clips "$QUALITY_CLIPS" --drop-fraction "$QUALITY_DROP" \
    --device "$DEVICE"
  if [[ ! -f $CORPUS/report.json ]]; then
    python "$HERE/scripts/build_corpus.py" --output "$CORPUS" "${SOURCES[@]}" --exclude-speakers "$CORPUS/all/excluded-speakers.txt"
  fi
fi

if step 4; then
  for split in validation train; do
    log "4/5 waveforms, spectra, pitch and energy: $split"
    tiny-voca-prepare --inputs "$CORPUS/$split.jsonl" --output "$CORPUS/features" --split "$split" \
      --config "$CORPUS/config.json" --workers "$WORKERS" --resume
  done
fi

if step 5; then
  # The released model's own phonemizer (dizione); its data comes with the released weights:
  #     hf download n4ze3m/tiny-voca --local-dir /workspace/weights   -> DIZIONE_DATA=/workspace/weights/phonemizer
  export TINY_VOCA_FRONTEND=${TINY_VOCA_FRONTEND:-own}
  if [[ $TINY_VOCA_FRONTEND == own && -z ${DIZIONE_DATA:-} ]]; then
    echo "set DIZIONE_DATA to the phonemizer folder of the released weights" >&2; exit 1
  fi
  # The aligner loads its phoneme recogniser offline, so fetch it once (cached under HF_HOME; keep that on the volume).
  python -c "from huggingface_hub import snapshot_download; from tiny_voca.align import ALIGNER_MODEL; print(snapshot_download(ALIGNER_MODEL))"
  REVISION=corpus-$(sha256sum "$CORPUS/train.jsonl" | cut -c1-12)
  log "5/5 phonemes and forced alignment ($REVISION)"
  tiny-voca-annotate --features "$CORPUS/features" --output "$CORPUS/annotated" --dataset-revision "$REVISION" \
    --config "$CORPUS/config.json" --device "$DEVICE" --splits train validation --resume
fi

log "done with steps $STEPS. corpus: $CORPUS/report.json, alignment: $CORPUS/annotated"
