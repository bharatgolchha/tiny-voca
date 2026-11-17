"""Make a blind A/B listening page from rendered samples.

    python scripts/ab_page.py --samples runs/vocoder/samples --first source_filter --second neural

Reads ``<samples>/clips.json`` and the wav files that ``tiny-voca-vocoder``
renders (``<samples>/<head>/<n>.wav`` and ``<samples>/recording/<n>.wav``) and
writes ``<samples>/ab.html``. For each clip the two heads are shown as A and B
in an order fixed by ``--seed`` and the clip, never by the head, with the
recording as a reference. The listener picks A, B or "same"; picks are kept in
the browser. Once every clip has a pick, "Show result" reveals which head won
how often. The pairs are copied under neutral names (``ab/<n>-a.wav``), but the
key is in the page, so a listener who decodes the source can find it: this is a
check for ourselves, not a formal test.
"""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil


def order(seed, file):
    return hashlib.sha256(f"{seed}:{file}".encode()).digest()[0] % 2 == 1


def page(clips, first, second, seed, samples):
    """The page, with each pair copied to ``ab/<n>-a.wav`` and ``ab/<n>-b.wav`` so file names do not tell."""
    rows, key = [], []
    (samples / "ab").mkdir(exist_ok=True)
    for n, clip in enumerate(clips, 1):
        swap = order(seed, clip["file"])
        a, b = (second, first) if swap else (first, second)
        key.append([a, b])
        for side, head in (("a", a), ("b", b)):
            shutil.copyfile(samples / head / clip["file"], samples / "ab" / f"{n:03d}-{side}.wav")
        audio = lambda path: f'<audio controls preload="none" src="{html.escape(path)}"></audio>'
        choice = "".join(f'<label><input type="radio" name="c{n}" value="{v}"> {label}</label> '
                         for v, label in (("A", "A better"), ("B", "B better"), ("=", "same")))
        rows.append(f'<tr><td>{n}</td><td><div class="muted">{html.escape(clip.get("text", ""))}</div>'
                    f'<div class="pair"><span>A</span>{audio(f"ab/{n:03d}-a.wav")}<span>B</span>{audio(f"ab/{n:03d}-b.wav")}</div>'
                    f'<details><summary class="muted">recording</summary>{audio("recording/" + clip["file"])}</details></td><td>{choice}</td></tr>')
    sealed = base64.b64encode(json.dumps(key).encode()).decode()
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind A/B listening</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#d6d3d1;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--accent:#93c5fd}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:960px;margin:0 auto;padding:24px 16px}
table{border-collapse:collapse;width:100%}td{border-bottom:1px solid var(--line);padding:10px 6px;vertical-align:top}
.muted{color:var(--muted);font-size:13px}.pair{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:6px 0}
audio{width:220px;max-width:100%}label{white-space:nowrap;display:inline-block;margin:2px 0}
button{font:inherit;padding:6px 14px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
#result{font-weight:600;color:var(--accent)}
</style></head><body><main><h1>Blind A/B listening</h1>
<p class="muted">Each row is the same sentence from two waveform generators, in random order. Pick the one that sounds more natural
and closer to a real person (use the recording as a reference), or "same" if you can't tell. Picks are saved in this browser.</p>
<table>""" + "\n".join(rows) + """</table>
<p><button id="reveal">Show result</button> <span id="count" class="muted"></span></p><p id="result"></p></main><script>
const key=JSON.parse(atob('""" + sealed + """'));const store='ab-""" + hashlib.sha256(f"{first}-{second}-{seed}".encode()).hexdigest()[:12] + """';
const load=()=>{try{return JSON.parse(localStorage.getItem(store)||'{}')}catch(e){return{}}};
const picks=load();const radios=[...document.querySelectorAll('input[type=radio]')];
radios.forEach(r=>{if(picks[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{picks[r.name]=r.value;
try{localStorage.setItem(store,JSON.stringify(picks))}catch(e){}count()})});
const count=()=>{document.getElementById('count').textContent=Object.keys(picks).length+' of '+key.length+' picked'};count();
document.getElementById('reveal').addEventListener('click',()=>{const out=document.getElementById('result');
if(Object.keys(picks).length<key.length){out.textContent='Pick every row first.';return}
const wins={};let same=0;key.forEach(([a,b],i)=>{const p=picks['c'+(i+1)];if(p==='=')same++;else{const w=p==='A'?a:b;wins[w]=(wins[w]||0)+1}});
out.textContent=Object.entries(wins).map(([h,n])=>h+' preferred '+n+' times').join(', ')+(same?', same '+same+' times':'')+'.'});
</script></body></html>
"""


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--first", default="source_filter")
    parser.add_argument("--second", default="neural")
    parser.add_argument("--seed", default="phase2")
    args = parser.parse_args(arguments)
    clips = json.loads((args.samples / "clips.json").read_text(encoding="utf-8"))
    for clip in clips:
        for folder in (args.first, args.second, "recording"):
            if not (args.samples / folder / clip["file"]).is_file():
                raise FileNotFoundError(args.samples / folder / clip["file"])
    (args.samples / "ab.html").write_text(page(clips, args.first, args.second, args.seed, args.samples), encoding="utf-8")
    print(args.samples / "ab.html")


if __name__ == "__main__":
    main()
