"""Pick random clips of the finished corpus for a listening check.

    python scripts/sample_corpus.py --annotated /workspace/data/corpus/annotated \
        --features /workspace/data/corpus/features --output /workspace/data/corpus/sample --clips 200

Writes ``<output>/index.html`` (open it in a browser: each clip with its text,
speaker and length, and a box to tick when something sounds wrong),
``<output>/audio/<id>.wav`` and ``<output>/summary.json`` (the annotation
summary and counts). ``--rejected`` more clips are taken from the clips the
aligner's checks threw out, so the reasons can be heard too. The choice
depends only on the clip ids and ``--seed``.
"""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.data import float_wave  # noqa: E402

RATE = 24000


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pick(rows, count, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())[:count]


def page(clips):
    rows = []
    for n, clip in enumerate(clips, 1):
        note = html.escape(clip.get("reasons", ""))
        rows.append(f"""<tr class="{'rejected' if note else ''}"><td>{n}</td><td><audio controls preload="none" src="audio/{html.escape(clip['file'])}"></audio></td>
<td><div>{html.escape(clip['text'])}</div><div class="muted">{html.escape(clip['speaker'])} · {clip['seconds']:.1f} s{(' · rejected: ' + note) if note else ''}</div></td>
<td><label><input type="checkbox" data-id="{html.escape(clip['id'])}"> wrong</label></td></tr>""")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corpus listening check</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#d6d3d1;--bad:#fef2f2}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--bad:#3f1d1d}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:960px;margin:0 auto;padding:24px 16px}
table{border-collapse:collapse;width:100%}td{border-bottom:1px solid var(--line);padding:8px 6px;vertical-align:top}
.muted{color:var(--muted);font-size:13px}tr.rejected{background:var(--bad)}audio{width:220px}
</style></head><body><main><h1>Corpus listening check</h1>
<p class="muted">Listen to a few dozen clips. Tick any where the audio is noisy, clipped, cut off, or doesn't match the text.
Ticks are saved in this browser; the count is at the bottom. Red rows are clips the checks rejected.</p>
<table>""" + "\n".join(rows) + """</table><p id="count" class="muted"></p></main><script>
const boxes=[...document.querySelectorAll('input[type=checkbox]')];
const load=()=>{try{return JSON.parse(localStorage.getItem('corpus-check')||'[]')}catch(e){return[]}};
const saved=new Set(load());boxes.forEach(b=>{b.checked=saved.has(b.dataset.id)});
const show=()=>{const ids=boxes.filter(b=>b.checked).map(b=>b.dataset.id);
try{localStorage.setItem('corpus-check',JSON.stringify(ids))}catch(e){}
document.getElementById('count').textContent=ids.length+' marked wrong'+(ids.length?': '+ids.join(', '):'')};
boxes.forEach(b=>b.addEventListener('change',show));show();
</script></body></html>
"""


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--rejected", type=int, default=20)
    parser.add_argument("--seed", default="phase1")
    args = parser.parse_args(arguments)
    accepted = read(args.annotated / f"{args.split}.jsonl")
    rejected_path = args.annotated / f"{args.split}.rejected.jsonl"
    rejected = read(rejected_path) if rejected_path.exists() else []
    paths = {r["id"]: r["path"] for r in read(args.features / f"{args.split}.jsonl")}
    (args.output / "audio").mkdir(parents=True, exist_ok=True)
    clips = []
    for row, reasons in [(r, "") for r in pick(accepted, args.clips, args.seed)] + \
                        [(r, "; ".join(r.get("reasons", []))) for r in pick(rejected, args.rejected, args.seed)]:
        stored = torch.load(args.features / paths[row["id"]], map_location="cpu", weights_only=True)
        wave = float_wave(stored["wave"]).numpy()
        name = hashlib.sha256(row["id"].encode()).hexdigest()[:16] + ".wav"
        sf.write(args.output / "audio" / name, wave, RATE, subtype="PCM_16")
        clips.append({"id": row["id"], "file": name, "text": row["text"], "speaker": row["speaker"],
                      "seconds": len(wave) / RATE, "reasons": reasons})
    (args.output / "index.html").write_text(page(clips), encoding="utf-8")
    summary_path = args.annotated / "annotations.json"
    annotation = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary = {"accepted": len(accepted), "rejected": len(rejected),
               "accepted_hours": round(sum(r["frames"] for r in accepted) * 256 / RATE / 3600, 2),
               "rejection_reasons": dict(Counter(reason.split(":")[0] for r in rejected for reason in r.get("reasons", []))),
               "complete": annotation.get("complete"), "counts": annotation.get("counts"), "sampled": len(clips)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
