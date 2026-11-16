"""Build the frozen test sets that need downloaded audio. Runs once, offline.

    python scripts/build_eval_sets.py --data-root ~/Projects/tiny-voca-data \
        --tiny-voca tiny-voca --libritts LibriTTS_R/test-clean

Inputs, both under ``--data-root``:

* ``--tiny-voca``: a download of the tiny-voca dataset's test split
  (``hf download n4ze3m/tiny-voca --repo-type dataset --include "data/test-*"``).
  Its sentences were held out of the released model's training, and its audio
  is what the model learned to imitate (Qwen3-TTS, voices F2 and M2).
* ``--libritts``: LibriTTS-R test-clean (CC BY 4.0, https://www.openslr.org/141/),
  human speakers that no training set of this project will contain.

Outputs:

* ``eval/sets/heldout.jsonl``: every test sentence of the tiny-voca dataset, with
  both voices' recordings written to ``<data-root>/tiny-voca-test/<voice>/<id>.wav``
* ``eval/sets/human.jsonl``: ``--human`` utterances of 3 to 12 seconds, spread over
  the test-clean speakers in a fixed order
* ``eval/sets/manifest.json``: counts, sources and the SHA-256 of every set file

The selection depends only on the inputs, so a rerun gives the same files.
``eval/sets/hard.jsonl`` is written by hand and not touched here.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import soundfile as sf

SETS = Path(__file__).resolve().parents[1] / "eval" / "sets"


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def heldout(data_root, source):
    import pyarrow.parquet as pq
    files = sorted((data_root / source).glob("data/test-*.parquet"))
    if not files:
        raise SystemExit(f"no data/test-*.parquet under {data_root / source}")
    sentences = {}
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if row.get("split", "test") != "test":
                    continue
                audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                relative = Path("tiny-voca-test") / row["voice_id"] / f"{row['text_id']}.wav"
                (data_root / relative).parent.mkdir(parents=True, exist_ok=True)
                sf.write(data_root / relative, audio, rate, subtype="PCM_16")
                entry = sentences.setdefault(row["text_id"], {"id": row["text_id"], "text": row["text"], "category": row.get("category"),
                                                             "origin": row.get("origin"), "source_license": row.get("source_license"),
                                                             "refs": {}})
                if entry["text"] != row["text"]:
                    raise SystemExit(f"two texts for sentence {row['text_id']}")
                entry["refs"][row["voice_id"]] = str(relative)
    rows = [s for _, s in sorted(sentences.items()) if len(s["refs"]) == 2]
    return rows, {"files": [f.name for f in files], "sentences_without_both_voices": len(sentences) - len(rows)}


def human(data_root, source, count, minimum, maximum, min_words, exclude):
    root = data_root / source
    candidates = {}
    for text_file in sorted(root.rglob("*.normalized.txt")):
        wav = text_file.with_name(text_file.name.replace(".normalized.txt", ".wav"))
        if not wav.exists():
            continue
        text = " ".join(text_file.read_text(encoding="utf-8").split())
        if len(text.split()) < min_words or text in exclude:
            continue
        duration = sf.info(wav).duration
        if not minimum <= duration <= maximum:
            continue
        utterance = wav.stem
        speaker = utterance.split("_")[0]
        candidates.setdefault(speaker, []).append({"id": f"ltr-{utterance}", "text": text, "speaker": speaker,
                                                   "ref": os.path.relpath(wav, data_root), "seconds": round(duration, 3)})
    if not candidates:
        raise SystemExit(f"no usable LibriTTS-R utterances under {root}")
    for speaker in candidates:
        candidates[speaker].sort(key=lambda r: stable(r["id"]))
    rows, depth = [], 0
    order = sorted(candidates, key=stable)
    while len(rows) < count and any(depth < len(candidates[s]) for s in order):
        for speaker in order:
            if depth < len(candidates[speaker]) and len(rows) < count:
                rows.append(candidates[speaker][depth])
        depth += 1
    rows.sort(key=lambda r: r["id"])
    return rows, {"speakers_available": len(candidates), "speakers_used": len({r["speaker"] for r in rows}),
                  "seconds": [minimum, maximum], "min_words": min_words}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tiny-voca", default="tiny-voca", help="Folder under --data-root with data/test-*.parquet")
    parser.add_argument("--libritts", default="LibriTTS_R/test-clean", help="Folder under --data-root with LibriTTS-R test-clean")
    parser.add_argument("--human", type=int, default=200)
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--min-words", type=int, default=6)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    hard = [json.loads(line) for line in (SETS / "hard.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    held, held_info = heldout(data_root, args.tiny_voca)
    exclude = {r["text"] for r in hard + held}
    people, people_info = human(data_root, args.libritts, args.human, args.min_seconds, args.max_seconds, args.min_words, exclude)
    write_rows(SETS / "heldout.jsonl", held)
    write_rows(SETS / "human.jsonl", people)
    manifest = {"sets": {}}
    for name, info, source in (("hard", {}, "written for this project"),
                               ("heldout", held_info, "n4ze3m/tiny-voca dataset, test split (CC BY 4.0)"),
                               ("human", people_info, "LibriTTS-R test-clean (CC BY 4.0), https://www.openslr.org/141/")):
        path = SETS / f"{name}.jsonl"
        manifest["sets"][name] = {"sentences": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
                                  "source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **info}
    (SETS / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
