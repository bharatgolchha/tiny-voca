"""Build the human-speech training corpus from LibriTTS-R and Hi-Fi TTS.

    python scripts/build_corpus.py --output /data/corpus \
        --libritts /data/LibriTTS_R/train-clean-100 /data/LibriTTS_R/train-clean-360 \
        --hifitts /data/hi_fi_tts_v0

Both sources are CC BY 4.0, so a model trained on them can be released or sold
with attribution. Non-commercial datasets are deliberately not supported.

What it does:

* LibriTTS-R (already 24 kHz): reads each ``<utterance>.wav`` with its
  ``<utterance>.normalized.txt``. The audio is used where it is.
* Hi-Fi TTS (44.1 kHz): reads ``<speaker>_manifest_<quality>_<split>.json``
  (``--hifitts-quality clean`` by default) and resamples each clip to 24 kHz
  16-bit FLAC under ``<output>/audio/hifi/``. Existing files are kept, so a
  rerun only does the missing ones.
* Drops clips shorter than ``--min-seconds`` or longer than ``--max-seconds``,
  with too few words, or whose text matches a sentence of the frozen test sets
  (``eval/sets/*.jsonl``), compared without case or punctuation.
* Holds out ``--validation-per-speaker`` clips of every speaker that has at
  least ``--min-validation-clips`` clips, chosen by a hash of the clip id among
  sentences read only once in the corpus (so no sentence is in both splits).
* Drops speakers with less than ``--min-speaker-minutes`` of kept audio.

Outputs, in ``--output``:

* ``train.jsonl`` and ``validation.jsonl``: rows for ``tiny_voca.prepare``
  (``id``, ``text``, ``voice_id``, ``audio`` as an absolute path, ``split``,
  ``seconds``, ``origin``, ``source_id``, ``source_license``, ``source_url``)
* ``speakers.json``: every kept speaker with its clips, hours and gender when
  the source lists it, in the order to use for ``Config.speakers``
* ``config.json``: the default model ``Config`` with these speakers, for
  ``tiny_voca.prepare --config`` and the later steps
* ``report.json``: counts, hours and the reasons clips were dropped

The selection depends only on the inputs, so a rerun gives the same files.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import soundfile as sf

SETS = Path(__file__).resolve().parents[1] / "eval" / "sets"
RATE = 24000
LICENSE = "CC BY 4.0"
LIBRITTS_URL = "https://www.openslr.org/141/"
HIFITTS_URL = "https://www.openslr.org/109/"


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def text_key(text):
    """Text compared without case, punctuation or spacing."""
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.lower()).split())


def eval_texts(folder):
    texts = set()
    for path in sorted(Path(folder).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.add(text_key(json.loads(line)["text"]))
    return texts


def libritts_genders(root):
    """LibriTTS-R ships speakers.tsv (READER, GENDER, ...) next to the subsets."""
    for folder in (root, root.parent):
        for name in ("speakers.tsv", "SPEAKERS.txt", "SPEAKERS.TXT"):
            path = folder / name
            if path.is_file():
                genders = {}
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith((";", "READER")) or not line.strip():
                        continue
                    parts = [p.strip() for p in re.split(r"\t|\|", line)]
                    if len(parts) >= 2 and parts[1] in ("F", "M"):
                        genders[parts[0]] = parts[1]
                return genders
    return {}


def libritts_clip(wav):
    text_file = wav.with_name(wav.stem + ".normalized.txt")
    if not text_file.is_file():
        return None
    info = sf.info(wav)
    return {"text": " ".join(text_file.read_text(encoding="utf-8").split()), "seconds": info.duration,
            "rate": info.samplerate, "channels": info.channels}


def libritts(roots, workers):
    rows, genders = [], {}
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"no LibriTTS-R folder at {root}")
        genders.update(libritts_genders(root))
        wavs = sorted(root.rglob("*.wav"))
        if not wavs:
            raise SystemExit(f"no .wav files under {root}")
        with ProcessPoolExecutor(workers) as pool:
            for wav, clip in zip(wavs, pool.map(libritts_clip, wavs, chunksize=256)):
                utterance = wav.stem
                speaker = utterance.split("_")[0]
                row = {"id": f"ltr-{utterance}", "voice_id": f"ltr-{speaker}", "audio": str(wav),
                       "origin": f"LibriTTS-R {root.name}", "source_id": utterance, "source_url": LIBRITTS_URL,
                       "gender": genders.get(speaker)}
                if clip is None:
                    row["drop"] = "no normalized text"
                elif clip["rate"] != RATE or clip["channels"] != 1:
                    row["drop"] = "not 24 kHz mono"
                else:
                    row.update(text=clip["text"], seconds=clip["seconds"])
                rows.append(row)
    return rows


def hifitts_manifests(root, quality):
    manifests = sorted(root.glob(f"*_manifest_{quality}_*.json"))
    if not manifests:
        raise SystemExit(f"no *_manifest_{quality}_*.json under {root}")
    for manifest in manifests:
        speaker = manifest.name.split("_manifest_")[0]
        split = manifest.stem.rsplit("_", 1)[1]
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                yield speaker, split, number, json.loads(line)


def resample(job):
    """Resample one Hi-Fi TTS clip to 24 kHz 16-bit FLAC; returns its duration or an error."""
    source, target = job
    target = Path(target)
    if target.is_file():
        return {"seconds": sf.info(target).duration}
    from scipy.signal import resample_poly
    try:
        wave, rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as error:      # unreadable source: report, keep going
        return {"error": f"{type(error).__name__}: {error}"}
    wave = wave.mean(axis=1)
    if rate != RATE:
        divisor = math.gcd(RATE, rate)
        wave = resample_poly(wave, RATE // divisor, rate // divisor).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    sf.write(temporary, np.clip(wave, -1.0, 1.0), RATE, subtype="PCM_16", format="FLAC")
    os.replace(temporary, target)
    return {"seconds": len(wave) / RATE}


def hifitts(root, output, quality, workers, keep):
    root = Path(root).expanduser().resolve()
    rows, jobs = [], []
    for speaker, split, number, entry in hifitts_manifests(root, quality):
        source = root / entry["audio_filepath"]
        stem = Path(entry["audio_filepath"]).with_suffix("")
        identifier = f"hifi-{speaker}-{stable(str(stem))[:12]}"
        text = " ".join(str(entry.get("text_normalized") or entry.get("text") or "").split())
        row = {"id": identifier, "voice_id": f"hifi-{speaker}", "text": text, "seconds": float(entry.get("duration", 0)),
               "origin": f"Hi-Fi TTS {quality} {split}", "source_id": str(entry["audio_filepath"]), "source_url": HIFITTS_URL,
               "gender": None, "audio": str(output / "audio" / "hifi" / speaker / f"{identifier}.flac")}
        if not text:
            row["drop"] = "no text"
        rows.append(row)
        if keep(row):
            jobs.append((row, (str(source), row["audio"])))
    with ProcessPoolExecutor(workers) as pool:
        for (row, _), result in zip(jobs, pool.map(resample, [j for _, j in jobs], chunksize=32)):
            if "error" in result:
                row["drop"] = "unreadable audio: " + result["error"]
            else:
                row["seconds"] = result["seconds"]
    return rows


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libritts", type=Path, nargs="*", default=[], help="LibriTTS-R subset folders, e.g. train-clean-100")
    parser.add_argument("--hifitts", type=Path, help="The hi_fi_tts_v0 folder with the *_manifest_*.json files")
    parser.add_argument("--hifitts-quality", default="clean", choices=("clean", "other"))
    parser.add_argument("--sets", type=Path, default=SETS, help="Folder of frozen test sets whose sentences are kept out")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--min-speaker-minutes", type=float, default=5.0)
    parser.add_argument("--validation-per-speaker", type=int, default=2)
    parser.add_argument("--min-validation-clips", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(arguments)
    if not args.libritts and not args.hifitts:
        parser.error("give --libritts, --hifitts or both")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    held_out = eval_texts(args.sets)

    def reason(row):
        if row.get("drop"):
            return row["drop"]
        if text_key(row["text"]) in held_out:
            return "sentence is in a test set"
        if len(row["text"].split()) < args.min_words:
            return "too few words"
        if row["seconds"] and not args.min_seconds <= row["seconds"] <= args.max_seconds:
            return "too short or too long"
        return None

    rows = libritts(args.libritts, args.workers)
    if args.hifitts:
        rows += hifitts(args.hifitts, output, args.hifitts_quality, args.workers, lambda r: reason(r) is None)
    dropped = Counter()
    kept = defaultdict(list)
    for row in rows:
        why = reason(row)
        if why is None and not args.min_seconds <= row["seconds"] <= args.max_seconds:
            why = "too short or too long"
        if why:
            dropped[why.split(":")[0]] += 1
            continue
        kept[row["voice_id"]].append(row)
    # Validation clips must not share a sentence with training (annotation refuses that leakage), so only
    # sentences read once in the whole corpus can be held out.
    readings = Counter(text_key(r["text"]) for clips in kept.values() for r in clips)
    seen_ids = set()
    speakers, train, validation = [], [], []
    for voice in sorted(kept):
        clips = sorted(kept[voice], key=lambda r: r["id"])
        minutes = sum(r["seconds"] for r in clips) / 60
        if minutes < args.min_speaker_minutes:
            dropped["speaker has too little audio"] += len(clips)
            continue
        held = set()
        if len(clips) >= args.min_validation_clips:
            unique = [r for r in clips if readings[text_key(r["text"])] == 1]
            held = {r["id"] for r in sorted(unique, key=lambda r: stable(r["id"]))[:args.validation_per_speaker]}
        for row in clips:
            if row["id"] in seen_ids:
                raise SystemExit(f"duplicate clip id {row['id']}")
            seen_ids.add(row["id"])
            split = "validation" if row["id"] in held else "train"
            out = {"id": row["id"], "text": row["text"], "voice_id": voice, "audio": row["audio"], "split": split,
                   "seconds": round(row["seconds"], 3), "origin": row["origin"], "source_id": row["source_id"],
                   "source_license": LICENSE, "source_url": row["source_url"]}
            (validation if split == "validation" else train).append(out)
        speakers.append({"id": voice, "source": voice.split("-")[0], "gender": clips[0].get("gender"), "clips": len(clips),
                         "hours": round(minutes / 60, 3)})
    for name, part in (("train", train), ("validation", validation)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
    (output / "speakers.json").write_text(json.dumps(speakers, indent=1) + "\n", encoding="utf-8")
    from tiny_voca.config import Config
    config = Config(speakers=tuple(s["id"] for s in speakers)).to_dict()
    (output / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    hours = defaultdict(float)
    for row in train + validation:
        hours[row["origin"]] += row["seconds"] / 3600
    report = {"clips": {"train": len(train), "validation": len(validation)},
              "hours": {"train": round(sum(r["seconds"] for r in train) / 3600, 2),
                        "validation": round(sum(r["seconds"] for r in validation) / 3600, 2),
                        "by_origin": {k: round(v, 2) for k, v in sorted(hours.items())}},
              "speakers": {"total": len(speakers), "by_source": dict(Counter(s["source"] for s in speakers)),
                           "by_gender": dict(Counter(str(s["gender"]) for s in speakers))},
              "dropped": dict(sorted(dropped.items())), "test_sentences_kept_out": len(held_out),
              "limits": {k: getattr(args, k) for k in ("min_seconds", "max_seconds", "min_words", "min_speaker_minutes",
                                                     "validation_per_speaker", "min_validation_clips", "hifitts_quality")}}
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
