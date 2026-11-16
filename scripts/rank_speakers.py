"""Rank the corpus speakers by recording quality and list the ones to leave out.

    python scripts/rank_speakers.py --corpus /data/corpus --clips 10 --drop-below 3.0

Scores ``--clips`` clips of every speaker in ``<corpus>/train.jsonl`` (chosen by a
hash of the clip id, so a rerun picks the same ones) with UTMOSv2, the
scoreboard's naturalness predictor. Writes, in ``--corpus``:

* ``speaker-quality.json``: every speaker with its mean, lowest and highest
  score, best first
* ``excluded-speakers.txt``: speakers whose mean is under ``--drop-below`` or
  in the bottom ``--drop-fraction``; give it to ``build_corpus.py
  --exclude-speakers`` and build the corpus again

Scores are kept in ``speaker-scores.jsonl`` as they are made, so an
interrupted run resumes. UTMOSv2 favours clean studio sound, which is what the
model should learn from; the listening check at the end of phase 1 decides
whether the threshold is right.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

import soundfile as sf


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sample(rows, clips):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["voice_id"]].append(row)
    return {speaker: sorted(members, key=lambda r: stable(r["id"]))[:clips] for speaker, members in sorted(by_speaker.items())}


def rank(scores, drop_below, drop_fraction):
    """Speakers best first, and the set to leave out."""
    table = []
    for speaker, values in scores.items():
        table.append({"speaker": speaker, "mean": round(statistics.fmean(values), 3), "lowest": round(min(values), 3),
                      "highest": round(max(values), 3), "clips": len(values)})
    table.sort(key=lambda r: (-r["mean"], r["speaker"]))
    dropped = {r["speaker"] for r in table if drop_below is not None and r["mean"] < drop_below}
    if drop_fraction:
        dropped |= {r["speaker"] for r in table[len(table) - int(len(table) * drop_fraction):]}
    return table, dropped


def main(arguments=None, scorer=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--drop-below", type=float, help="Leave out speakers whose mean score is under this")
    parser.add_argument("--drop-fraction", type=float, default=0.0, help="Leave out this fraction of speakers, worst first")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(arguments)
    rows = [json.loads(line) for line in (args.corpus / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = sample(rows, args.clips)
    journal = args.corpus / "speaker-scores.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record["utmos"]
    if scorer is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        import torch
        from tiny_voca.cpus import available_cpus
        from tiny_voca.scoreboard import Naturalness
        torch.set_num_threads(available_cpus())      # not the host's core count: that thrashes in a container
        scorer = Naturalness(args.device)
    pending = [(speaker, row) for speaker, members in chosen.items() for row in members if row["id"] not in done]
    print(json.dumps({"speakers": len(chosen), "clips_to_score": len(pending), "already_scored": len(done)}), flush=True)
    with tempfile.TemporaryDirectory() as folder, journal.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), 64):
            batch = pending[start:start + 64]
            paths = []
            for n, (speaker, row) in enumerate(batch):        # one folder of wav copies per batch: UTMOSv2 scores a folder at once
                wave, rate = sf.read(row["audio"], dtype="float32")
                path = Path(folder) / f"{start + n:07d}.wav"
                sf.write(path, wave, rate, subtype="PCM_16")
                paths.append(path)
            for (speaker, row), (path, value) in zip(batch, scorer(paths).items()):
                done[row["id"]] = value
                stream.write(json.dumps({"id": row["id"], "speaker": speaker, "utmos": round(value, 4)}) + "\n")
                path.unlink()
            stream.flush()
            print(json.dumps({"scored": start + len(batch), "of": len(pending)}), flush=True)
    scores = {speaker: [done[r["id"]] for r in members] for speaker, members in chosen.items()}
    table, dropped = rank(scores, args.drop_below, args.drop_fraction)
    for row in table:
        row["excluded"] = row["speaker"] in dropped
    (args.corpus / "speaker-quality.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    (args.corpus / "excluded-speakers.txt").write_text("".join(s + "\n" for s in sorted(dropped)), encoding="utf-8")
    summary = {"speakers": len(table), "excluded": len(dropped),
               "mean_of_kept": round(statistics.fmean(r["mean"] for r in table if not r["excluded"]), 3) if len(dropped) < len(table) else None,
               "best": table[:3], "worst": table[-3:]}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
