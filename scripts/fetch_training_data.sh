#!/usr/bin/env bash
# Download and unpack the human-speech training data on the training machine.
#
#     DATA=/workspace/data bash scripts/fetch_training_data.sh
#
# LibriTTS-R train-clean-100 and train-clean-360 (openslr 141) and Hi-Fi TTS
# (openslr 109), about 78 GB of archives, all CC BY 4.0. Downloads resume when
# run again; an archive is deleted only after it unpacked cleanly. Set MIRROR to
# another openslr mirror (for example https://openslr.elda.org or
# https://openslr.trmal.net) when the main site is slow.
set -euo pipefail

DATA=${DATA:?set DATA to the folder for the training data}
MIRROR=${MIRROR:-https://www.openslr.org}
MIN_FREE_GB=${MIN_FREE_GB:-200}
mkdir -p "$DATA"
cd "$DATA"

free_gb() { df -Pk . | awk 'NR==2 {print int($4 / 1048576)}'; }

fetch() {        # fetch <openslr number> <archive>
  local number=$1 archive=$2 done_marker=".unpacked-$2"
  if [[ -f $done_marker ]]; then echo "$archive: already unpacked"; return; fi
  if (( $(free_gb) < MIN_FREE_GB )); then
    echo "only $(free_gb) GB free, need $MIN_FREE_GB GB: stopping" >&2; exit 1
  fi
  echo "$archive: downloading from $MIRROR"
  curl -fL --retry 10 --retry-delay 5 -C - -o "$archive" "$MIRROR/resources/$number/$archive"
  echo "$archive: unpacking"
  tar -xzf "$archive"
  touch "$done_marker"
  rm -f "$archive"
}

fetch 141 train_clean_100.tar.gz
fetch 141 train_clean_360.tar.gz
fetch 109 hi_fi_tts_v0.tar.gz

echo "done: $(du -sh LibriTTS_R hi_fi_tts_v0 2>/dev/null | tr '\n' ' ')"
echo "LibriTTS-R clips: $(find LibriTTS_R -name '*.wav' | wc -l)"
echo "Hi-Fi TTS manifests: $(ls hi_fi_tts_v0/*_manifest_*.json | wc -l)"
