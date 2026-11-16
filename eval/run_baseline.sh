#!/usr/bin/env bash
# Phase 0 baseline: score today's tiny-voca, the audio it learned from, human
# speech and Kokoro-82M on the same frozen sentences. Run from the repo root in
# the scoring environment (.venv-eval). Every step resumes where it stopped.
#
#   DATA=~/Projects/tiny-voca-data bash eval/run_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-$HOME/Projects/tiny-voca-data}
RUN=${RUN:-runs/baseline}
LIBRITTS=${LIBRITTS:-LibriTTS_R/test-clean}
CHECKPOINT=${CHECKPOINT:-weights/tiny-voca.pt}
export DIZIONE_DATA=${DIZIONE_DATA:-weights/phonemizer}
SETS=(eval/sets/heldout.jsonl eval/sets/hard.jsonl eval/sets/human.jsonl)
board() { python -m tiny_voca.scoreboard "$@"; }

[ -s eval/sets/human.jsonl ] || python scripts/build_eval_sets.py --data-root "$DATA" --tiny-voca tiny-voca --libritts "$LIBRITTS"
board synth --run "$RUN" --system tiny-voca --checkpoint "$CHECKPOINT" --voices F2 M2 --set "${SETS[@]}"
board synth --run "$RUN" --system kokoro --voices af_heart am_michael --set "${SETS[@]}"
board refs --run "$RUN" --system recordings --set eval/sets/heldout.jsonl --data-root "$DATA"
board refs --run "$RUN" --system human --set eval/sets/human.jsonl --data-root "$DATA"
board bench --run "$RUN" --checkpoint "$CHECKPOINT"
board score --run "$RUN" --metrics asr voice utmos
board report --run "$RUN" --reference tiny-voca:F2=recordings:F2 tiny-voca:M2=recordings:M2
board pairs --run "$RUN" --a tiny-voca:M2 --b kokoro:am_michael --name m2-vs-kokoro
board pairs --run "$RUN" --a tiny-voca:F2 --b recordings:F2 --set heldout --name f2-vs-its-training-audio
