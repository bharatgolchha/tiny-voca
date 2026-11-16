# Scoreboard

Every change to tiny-voca is judged on the same frozen sentences, with the same
scores, against the same reference points. The code is
`src/tiny_voca/scoreboard.py`; `eval/run_baseline.sh` runs the whole baseline.

## Test sets

| Set | Sentences | What it checks | Source |
| --- | ---: | --- | --- |
| `sets/hard.jsonl` | 200 | Numbers, money, dates, times, codes, abbreviations, names, homographs, questions, exclamations, long sentences, punctuation | Written for this project |
| `sets/heldout.jsonl` | about 300 | Ordinary sentences the released model never trained on, with its training audio (Qwen3-TTS, F2 and M2) for comparison | tiny-voca dataset, test split (CC BY 4.0) |
| `sets/human.jsonl` | 200 | Real people reading, the bar to aim for | LibriTTS-R test-clean (CC BY 4.0) |

`heldout` and `human` are built by `scripts/build_eval_sets.py` from downloaded
audio; the selection is fixed, and `sets/manifest.json` records each file's
SHA-256. Do not edit a set once scores exist for it: add a new set instead.

No sentence of these sets may be in a training set. Phase 1 data preparation
must drop every training clip whose text matches one of them, because some
`heldout` sentences are LibriTTS-R transcripts.

## Scores

| Score | Model | Meaning |
| --- | --- | --- |
| Naturalness | UTMOSv2 | Predicted mean opinion score, 1 to 5 |
| Word errors | Whisper large-v3-turbo (mlx-whisper on a Mac, faster-whisper elsewhere) | Share of words misheard, after Whisper's English normalisation |
| Voice similarity | WavLM x-vector | Cosine to the centroid of the target voice's recordings |
| Voice consistency | WavLM x-vector | Cosine of each clip to the voice's other clips |
| Times real time | | Seconds of audio made per second of compute, from synthesis timing |

Scores steer day to day. A blind A/B listening test (`web/ab/index.html`, made
by `scoreboard pairs`) decides each phase gate: 23 or more wins out of 30 is a
preference with under 1% chance of being luck.

## Running

```bash
python3.12 -m venv .venv-eval && source .venv-eval/bin/activate
pip install -e ".[own,prepare,score]" "git+https://github.com/sarulab-speech/UTMOSv2.git"
DATA=~/Projects/tiny-voca-data bash eval/run_baseline.sh
```

`runs/<name>/report.md` holds the table. To listen:

```bash
cd runs/baseline && python -m http.server 8765
# open http://localhost:8765/ab/?test=m2-vs-kokoro
```
