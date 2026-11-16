"""The scoreboard: one way to score every text-to-speech system on the same sentences.

    python -m tiny_voca.scoreboard synth  --run runs/baseline --system tiny-voca --checkpoint weights/tiny-voca.pt ...
    python -m tiny_voca.scoreboard synth  --run runs/baseline --system kokoro --voices af_heart am_michael ...
    python -m tiny_voca.scoreboard refs   --run runs/baseline --system recordings --set eval/sets/heldout.jsonl ...
    python -m tiny_voca.scoreboard score  --run runs/baseline --metrics utmos asr voice
    python -m tiny_voca.scoreboard report --run runs/baseline --reference tiny-voca:F2=recordings:F2 ...
    python -m tiny_voca.scoreboard pairs  --run runs/baseline --a tiny-voca:M2 --b kokoro:am_michael --set heldout
    python -m tiny_voca.scoreboard bench  --run runs/baseline --checkpoint weights/tiny-voca.pt --threads 1 4

A run directory holds every clip it scores, so a run can be listened to,
re-scored and compared later:

    clips.jsonl            one row per clip: system, voice, set, sentence id, text, audio path, timing
    audio/<system>/<voice>/<set>/<id>.wav
    scores-<metric>.jsonl  one row per clip and metric; scoring resumes where it stopped
    report.json, report.md the summary
    ab/                    the blind A/B listening page and its pairs
    bench.json             speed and memory of the model on this machine

Metrics (each loads its model only when asked for):

* ``utmos``  predicted naturalness, UTMOSv2 (a 1 to 5 mean-opinion-score estimate)
* ``asr``    transcript and word errors, Whisper (mlx-whisper on Apple silicon, else faster-whisper)
* ``voice``  speaker embedding, WavLM x-vector; the report turns it into similarity scores

The scoring models judge the audio. None of them is part of the TTS model.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

SETS_DIR = Path(__file__).resolve().parents[2] / "eval" / "sets"
AB_TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "ab" / "index.html"
METRICS = ("utmos", "asr", "voice")
SCORE_RATE = 16000


# ---------------------------------------------------------------- run files

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def clip_key(system, voice, set_name, sentence):
    return f"{system}/{voice}/{set_name}/{sentence}"


def set_name(path):
    return Path(path).stem


def load_set(path, limit=None):
    """Sentences of a frozen test set: rows with ``id`` and ``text``; reference
    sets also carry ``refs`` ({voice: audio path}) or ``ref`` and ``speaker``."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty or missing test set: {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate sentence ids in {path}")
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"sentence {row.get('id')} in {path} has no text")
    return rows[:limit] if limit else rows


def load_clips(run):
    """Latest row per clip key (a re-synthesised clip replaces the old row)."""
    clips = {}
    for row in read_jsonl(Path(run) / "clips.jsonl"):
        clips[row["key"]] = row
    return clips


def load_scores(run, metric):
    scores = {}
    for row in read_jsonl(Path(run) / f"scores-{metric}.jsonl"):
        scores[row["key"]] = row
    return scores


def audio_path(run, system, voice, set_name_, sentence):
    return Path(run) / "audio" / system / voice / set_name_ / f"{sentence}.wav"


def write_wave(path, wave, rate):
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(wave, dtype=np.float32)
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak > 0.99:
        wave = wave * (0.99 / peak)
    sf.write(path, wave, rate, subtype="PCM_16")
    return len(wave) / rate


def load_16k(path):
    """Mono float32 audio at 16 kHz, the rate every scoring model expects."""
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SCORE_RATE:
        from scipy.signal import resample_poly
        factor = math.gcd(rate, SCORE_RATE)
        audio = resample_poly(audio, SCORE_RATE // factor, rate // factor).astype(np.float32)
    return audio


# ---------------------------------------------------------------- synthesis

class TinyVocaSystem:
    rate = 24000

    def __init__(self, checkpoint, device="cpu", seed=0):
        from .infer import load_model, match_frontend
        if "TINY_VOCA_FRONTEND" not in os.environ:
            match_frontend(checkpoint)
        self.model = load_model(checkpoint, device)
        self.seed = seed
        self.voices = self.model.config.speakers

    def __call__(self, text, voice):
        from .infer import speak_text
        wave, _ = speak_text(self.model, text, voice, seed=self.seed)
        return wave.numpy()


class KokoroSystem:
    """Kokoro-82M (Apache-2.0) as a reference point: ``pip install kokoro``."""
    rate = 24000

    def __init__(self, lang_code="a"):
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voices = None

    def __call__(self, text, voice):
        import numpy as np
        pieces = [np.asarray(result.audio, dtype=np.float32) for result in self.pipeline(text, voice=voice) if result.audio is not None]
        if not pieces:
            raise ValueError("kokoro produced no audio")
        return np.concatenate(pieces)


def make_system(args):
    if args.system == "tiny-voca" or args.checkpoint:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for a tiny-voca system")
        return TinyVocaSystem(args.checkpoint, args.device, args.seed)
    if args.system == "kokoro":
        return KokoroSystem()
    raise SystemExit(f"unknown system {args.system!r}: use tiny-voca (with --checkpoint) or kokoro")


def synth(args):
    import torch
    torch.set_num_threads(args.threads)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    system = make_system(args)
    name = args.name or args.system
    voices = args.voices or list(system.voices or ())
    if not voices:
        raise SystemExit("--voices is required for this system")
    done = load_clips(run) if not args.overwrite else {}
    made = failed = 0
    for path in args.set:
        rows = load_set(path, args.limit)
        for voice in voices:
            for row in rows:
                key = clip_key(name, voice, set_name(path), row["id"])
                target = audio_path(run, name, voice, set_name(path), row["id"])
                if key in done and target.exists():
                    continue
                started = time.perf_counter()
                try:
                    wave = system(row["text"], voice)
                except Exception as error:      # one unreadable sentence must not stop a long run
                    append_jsonl(run / "failures.jsonl", [{"key": key, "text": row["text"], "error": f"{type(error).__name__}: {error}"}])
                    failed += 1
                    continue
                seconds = time.perf_counter() - started
                duration = write_wave(target, wave, system.rate)
                append_jsonl(run / "clips.jsonl", [{
                    "key": key, "system": name, "voice": voice, "set": set_name(path), "id": row["id"], "text": row["text"],
                    "category": row.get("category"), "path": str(target.relative_to(run)), "audio_seconds": round(duration, 4),
                    "synth_seconds": round(seconds, 4), "threads": args.threads}])
                made += 1
                if made % 25 == 0:
                    print(json.dumps({"system": name, "made": made, "last": key}), flush=True)
    print(json.dumps({"system": name, "made": made, "failed": failed}))


def refs(args):
    """Add recordings (the training audio, or human speech) as a system of their
    own. The files are copied into the run so the run stands alone."""
    import soundfile as sf
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    rows = load_set(args.set, args.limit)
    name, added = set_name(args.set), 0
    done = load_clips(run)
    new = []
    for row in rows:
        if "refs" in row:
            sources = [(voice, root / path, None) for voice, path in sorted(row["refs"].items())]
        elif "ref" in row:
            sources = [(args.voice, root / row["ref"], row.get("speaker"))]
        else:
            raise ValueError(f"sentence {row['id']} has no reference audio")
        for voice, source, speaker in sources:
            key = clip_key(args.system, voice, name, row["id"])
            target = audio_path(run, args.system, voice, name, row["id"])
            if key in done and target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            new.append({"key": key, "system": args.system, "voice": voice, "set": name, "id": row["id"], "text": row["text"],
                        "category": row.get("category"), "speaker": speaker, "path": str(target.relative_to(run)),
                        "audio_seconds": round(sf.info(target).duration, 4)})
            added += 1
    append_jsonl(run / "clips.jsonl", new)
    print(json.dumps({"system": args.system, "set": name, "added": added}))


# ---------------------------------------------------------------- scoring

def text_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        return EnglishTextNormalizer({})
    except Exception:
        return lambda text: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def apple_silicon():
    import platform
    return sys.platform == "darwin" and platform.machine() == "arm64"


class Transcriber:
    def __init__(self, backend, model):
        self.backend = backend or ("mlx" if apple_silicon() else "faster-whisper")
        if self.backend == "mlx":
            import mlx_whisper
            self.model = model or "mlx-community/whisper-large-v3-turbo"
            self._run = lambda audio: mlx_whisper.transcribe(audio, path_or_hf_repo=self.model, language="en", temperature=0.0,
                                                             condition_on_previous_text=False)["text"]
        elif self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            self.model = model or "large-v3"
            asr = WhisperModel(self.model, device="auto", compute_type="int8")

            def run(audio):
                segments, _ = asr.transcribe(audio, language="en", beam_size=5, condition_on_previous_text=False, temperature=0.0)
                return " ".join(segment.text.strip() for segment in segments)
            self._run = run
        else:
            raise SystemExit(f"unknown ASR backend {backend!r}")

    @property
    def name(self):
        return f"{self.backend}:{self.model}"

    def __call__(self, path):
        return self._run(load_16k(path)).strip()


class Naturalness:
    def __init__(self, device):
        import utmosv2
        self.device = device
        self.model = utmosv2.create_model(pretrained=True)

    def _predict(self, **options):
        try:
            return self.model.predict(device=self.device, **options)
        except (RuntimeError, NotImplementedError) as error:
            if self.device == "cpu":
                raise
            print(json.dumps({"utmos": f"{self.device} failed ({type(error).__name__}), using the CPU"}), flush=True)
            self.device = "cpu"
            return self.model.predict(device=self.device, **options)

    def __call__(self, paths):
        """Predicted MOS for each path. Whole folders go in one call when the
        installed UTMOSv2 accepts it, which is much faster."""
        out = {}
        folders = defaultdict(list)
        for path in paths:
            folders[Path(path).parent].append(Path(path))
        for folder, members in folders.items():
            wanted = {p.resolve() for p in members}
            try:
                results = self._predict(input_dir=str(folder), num_workers=0, verbose=False)
                for result in results:
                    resolved = Path(result["file_path"]).resolve()
                    if resolved in wanted:
                        out[resolved] = float(result["predicted_mos"])
            except TypeError:
                pass
            for path in members:
                if path.resolve() not in out:
                    try:
                        value = self._predict(input_path=str(path), verbose=False)
                    except TypeError:
                        value = self.model.predict(input_path=str(path))
                    out[path.resolve()] = float(value[0]["predicted_mos"] if isinstance(value, list) else value)
        return {path: out[Path(path).resolve()] for path in paths}


class VoiceEmbedder:
    MODEL = "microsoft/wavlm-base-plus-sv"

    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch, self.device = torch, device
        self.extractor = AutoFeatureExtractor.from_pretrained(self.MODEL)
        self.model = WavLMForXVector.from_pretrained(self.MODEL).to(device).eval()

    def __call__(self, path):
        inputs = self.extractor(load_16k(path), sampling_rate=SCORE_RATE, return_tensors="pt")
        with self.torch.no_grad():
            embedding = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings[0]
        embedding = self.torch.nn.functional.normalize(embedding, dim=-1)
        return [round(float(x), 5) for x in embedding.cpu()]


def score(args):
    run = Path(args.run)
    clips = load_clips(run)
    if args.system:
        clips = {k: v for k, v in clips.items() if v["system"] in args.system}
    for metric in args.metrics:
        done = load_scores(run, metric)
        todo = [clip for key, clip in sorted(clips.items()) if key not in done or args.overwrite]
        print(json.dumps({"metric": metric, "clips": len(clips), "to_score": len(todo)}), flush=True)
        if not todo:
            continue
        started = time.time()
        if metric == "utmos":
            scorer = Naturalness(args.device)
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start:start + args.chunk]
                values = scorer([run / clip["path"] for clip in chunk])
                append_jsonl(run / "scores-utmos.jsonl", [{"key": clip["key"], "utmos": round(values[run / clip["path"]], 4)} for clip in chunk])
                print(json.dumps({"metric": metric, "scored": start + len(chunk), "of": len(todo),
                                  "seconds": round(time.time() - started)}), flush=True)
        elif metric == "asr":
            scorer = Transcriber(args.asr_backend, args.asr_model)
            normalize = text_normalizer()
            for number, clip in enumerate(todo, 1):
                transcript = scorer(run / clip["path"])
                errors, words = word_errors(normalize(clip["text"]), normalize(transcript))
                append_jsonl(run / "scores-asr.jsonl", [{"key": clip["key"], "asr_model": scorer.name, "transcript": transcript,
                                                         "errors": errors, "words": words}])
                if number % 50 == 0:
                    print(json.dumps({"metric": metric, "scored": number, "of": len(todo), "seconds": round(time.time() - started)}), flush=True)
        elif metric == "voice":
            scorer = VoiceEmbedder(args.device if args.device != "mps" else "cpu")
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start:start + args.chunk]
                append_jsonl(run / "scores-voice.jsonl", [{"key": clip["key"], "embedding": scorer(run / clip["path"])} for clip in chunk])
        else:
            raise SystemExit(f"unknown metric {metric!r}")
        print(json.dumps({"metric": metric, "done": len(todo), "seconds": round(time.time() - started)}), flush=True)


def word_errors(reference, hypothesis):
    """(substitutions + deletions + insertions, reference words) on normalized text."""
    import jiwer
    reference = reference.strip()
    if not reference:
        return 0, 0
    output = jiwer.process_words(reference, hypothesis.strip() or "<empty>")
    return output.substitutions + output.deletions + output.insertions, len(reference.split())


# ---------------------------------------------------------------- report

def mean_ci(values):
    """Mean and the half width of its 95% interval (normal approximation)."""
    values = [float(v) for v in values]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(variance / len(values))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def centroid(vectors):
    size = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(size)]


def consistency(vectors):
    """Mean cosine of each clip to the centroid of the OTHER clips of the same
    voice: how stable the voice is from sentence to sentence."""
    if len(vectors) < 2:
        return None
    total = [sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
    values = []
    for v in vectors:
        rest = [(t - x) / (len(vectors) - 1) for t, x in zip(total, v)]
        values.append(cosine(v, rest))
    return sum(values) / len(values)


def sign_test(wins, losses):
    """One-sided p-value that ``wins`` out of ``wins + losses`` forced choices
    happen by chance (ties are left out)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def parse_references(items):
    """``system:voice=refsystem:refvoice`` pairs."""
    mapping = {}
    for item in items or ():
        left, _, right = item.partition("=")
        if ":" not in left or ":" not in right:
            raise SystemExit(f"--reference wants system:voice=refsystem:refvoice, not {item!r}")
        mapping[tuple(left.split(":", 1))] = tuple(right.split(":", 1))
    return mapping


def build_report(run, references=()):
    run = Path(run)
    clips = load_clips(run)
    if not clips:
        raise SystemExit("this run has no clips yet")
    utmos, asr, voice = load_scores(run, "utmos"), load_scores(run, "asr"), load_scores(run, "voice")
    references = parse_references(references)
    groups = defaultdict(list)
    for clip in clips.values():
        groups[(clip["system"], clip["voice"], clip["set"])].append(clip)
    # Voice embeddings per system and voice, over all sets; per speaker for multi-speaker recordings.
    voices = defaultdict(list)
    for key, clip in clips.items():
        if key in voice:
            voices[(clip["system"], clip["voice"], clip.get("speaker"))].append(voice[key]["embedding"])
    asr_models = sorted({row["asr_model"] for row in asr.values() if "asr_model" in row})
    rows = []
    for (system, voice_name, set_name_), members in sorted(groups.items()):
        row = {"system": system, "voice": voice_name, "set": set_name_, "clips": len(members),
               "audio_seconds": round(sum(c["audio_seconds"] for c in members), 1)}
        values = [utmos[c["key"]]["utmos"] for c in members if c["key"] in utmos]
        if values:
            mean, half = mean_ci(values)
            row.update(utmos=round(mean, 3), utmos_ci=round(half, 3) if half is not None else None, utmos_clips=len(values))
        scored = [asr[c["key"]] for c in members if c["key"] in asr]
        if scored:
            errors, words = sum(s["errors"] for s in scored), sum(s["words"] for s in scored)
            row.update(wer=round(errors / max(words, 1), 4), wer_words=words,
                       clips_with_errors=sum(1 for s in scored if s["errors"]))
            categories = defaultdict(lambda: [0, 0])
            for clip in members:
                if clip["key"] in asr and clip.get("category"):
                    categories[clip["category"]][0] += asr[clip["key"]]["errors"]
                    categories[clip["category"]][1] += asr[clip["key"]]["words"]
            if categories:
                row["wer_by_category"] = {k: round(e / max(w, 1), 4) for k, (e, w) in sorted(categories.items())}
        timed = [c for c in members if c.get("synth_seconds") is not None]
        if timed:
            row["real_time_factor"] = round(sum(c["synth_seconds"] for c in timed) / max(sum(c["audio_seconds"] for c in timed), 1e-9), 4)
            row["times_real_time"] = round(1 / max(row["real_time_factor"], 1e-9), 1)
        embedded = [voice[c["key"]]["embedding"] for c in members if c["key"] in voice]
        if embedded:
            speakers = {c.get("speaker") for c in members}
            stable = [consistency(voices[(system, voice_name, s)]) for s in speakers]
            stable = [s for s in stable if s is not None]
            if stable:
                row["voice_consistency"] = round(sum(stable) / len(stable), 4)
            target = references.get((system, voice_name))
            if target and voices.get((target[0], target[1], None)):
                reference = centroid(voices[(target[0], target[1], None)])
                row["voice_similarity"] = round(sum(cosine(e, reference) for e in embedded) / len(embedded), 4)
                row["voice_reference"] = f"{target[0]}:{target[1]}"
        rows.append(row)
    report = {"run": str(run), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "asr_models": asr_models,
              "naturalness_model": "UTMOSv2" if utmos else None, "voice_model": VoiceEmbedder.MODEL if voice else None,
              "rows": rows}
    failures = read_jsonl(run / "failures.jsonl")
    if failures:
        report["failures"] = [f for f in failures if f["key"] not in clips]
    bench = run / "bench.json"
    if bench.exists():
        report["bench"] = json.loads(bench.read_text())
    return report


def report_markdown(report):
    def cell(value, scale=1, digits=2, suffix=""):
        return "" if value is None else f"{value * scale:.{digits}f}{suffix}"
    lines = [f"# Scoreboard: {report['run']}", "", f"Made {report['created']}. ASR: {', '.join(report['asr_models']) or 'none'}. "
             f"Naturalness: {report['naturalness_model'] or 'none'}. Voice: {report['voice_model'] or 'none'}.", "",
             "| System | Voice | Set | Clips | Naturalness (UTMOSv2) | Word errors | Voice similarity | Voice consistency | Times real time |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["rows"]:
        mos = cell(row.get("utmos"))
        if row.get("utmos_ci") is not None:
            mos += f" ± {row['utmos_ci']:.2f}"
        lines.append(f"| {row['system']} | {row['voice']} | {row['set']} | {row['clips']} | {mos} | {cell(row.get('wer'), 100, 1, '%')} | "
                     f"{cell(row.get('voice_similarity'), digits=3)} | {cell(row.get('voice_consistency'), digits=3)} | "
                     f"{cell(row.get('times_real_time'), digits=0)} |")
    categories = [r for r in report["rows"] if r.get("wer_by_category")]
    if categories:
        names = sorted({name for r in categories for name in r["wer_by_category"]})
        lines += ["", "## Word errors by kind of sentence", "", "| System | Voice | " + " | ".join(names) + " |",
                  "| --- | --- | " + " | ".join("---:" for _ in names) + " |"]
        for r in categories:
            lines.append(f"| {r['system']} | {r['voice']} | " + " | ".join(cell(r['wer_by_category'].get(n), 100, 1, '%') for n in names) + " |")
    if report.get("failures"):
        lines += ["", f"## Sentences a system could not read ({len(report['failures'])})", ""]
        lines += [f"- `{f['key']}`: {f['error']}" for f in report["failures"][:50]]
    if report.get("bench"):
        lines += ["", "## Speed on this machine", "", "```json", json.dumps(report["bench"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def report_command(args):
    report = build_report(args.run, args.reference)
    run = Path(args.run)
    (run / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (run / "report.md").write_text(report_markdown(report), encoding="utf-8")
    print(report_markdown(report))


# ---------------------------------------------------------------- A/B page

def make_pairs(clips, a, b, set_filter, count, seed):
    """Blind forced-choice pairs of the same sentence from systems ``a`` and ``b``
    (each ``system:voice``). The side of each system is random, half and half."""
    def index(spec):
        system, _, voice = spec.partition(":")
        return {(c["set"], c["id"]): c for c in clips.values()
                if c["system"] == system and c["voice"] == voice and (not set_filter or c["set"] in set_filter)}
    left, right = index(a), index(b)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise SystemExit(f"no sentence was made by both {a} and {b}")
    generator = random.Random(seed)
    chosen = generator.sample(shared, min(count, len(shared)))
    sides = [i % 2 == 0 for i in range(len(chosen))]
    generator.shuffle(sides)
    pairs = []
    for number, (key, a_first) in enumerate(zip(chosen, sides), 1):
        first, second = (left[key], right[key]) if a_first else (right[key], left[key])
        pairs.append({"n": number, "set": key[0], "id": key[1], "text": first["text"],
                      "first": "../" + first["path"], "second": "../" + second["path"],
                      "first_is": "a" if a_first else "b"})
    return {"a": a, "b": b, "seed": seed, "pairs": pairs}


def pairs_command(args):
    run = Path(args.run)
    test = make_pairs(load_clips(run), args.a, args.b, args.set, args.count, args.seed)
    folder = run / "ab"
    folder.mkdir(exist_ok=True)
    name = args.name or f"{args.a}-vs-{args.b}".replace(":", "_")
    (folder / f"{name}.json").write_text(json.dumps(test, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(AB_TEMPLATE, folder / "index.html")
    print(f"Wrote {len(test['pairs'])} pairs. To listen:\n  cd {run} && python -m http.server 8765\n"
          f"  then open http://localhost:8765/ab/?test={name}")


# ---------------------------------------------------------------- speed

def peak_memory_mb():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def bench(args):
    """Speed of the tiny-voca checkpoint on this machine: load time, how many
    times faster than real time per thread count, time to the first sentence of
    a paragraph, peak memory."""
    import torch
    from .infer import finished, sentences, speak, speak_text
    texts = [row["text"] for row in load_set(args.set, args.sentences)]
    started = time.perf_counter()
    system = TinyVocaSystem(args.checkpoint, "cpu")
    result = {"checkpoint": str(args.checkpoint), "load_seconds": round(time.perf_counter() - started, 3),
              "parameters": sum(p.numel() for p in system.model.parameters()), "sentences": len(texts), "threads": {}}
    voice = args.voice or system.voices[0]
    for threads in args.threads:
        torch.set_num_threads(threads)
        speak_text(system.model, texts[0], voice)                       # warm-up
        audio = compute = 0.0
        for text in texts:
            begun = time.perf_counter()
            wave, _ = speak_text(system.model, text, voice)
            compute += time.perf_counter() - begun
            audio += len(wave) / system.rate
        paragraph = " ".join(texts[:3])
        begun = time.perf_counter()
        speak(system.model, sentences(finished(paragraph))[0], voice)
        first = time.perf_counter() - begun
        result["threads"][str(threads)] = {"times_real_time": round(audio / max(compute, 1e-9), 1),
                                           "seconds_per_audio_second": round(compute / max(audio, 1e-9), 4),
                                           "first_sentence_seconds": round(first, 3)}
    result["peak_memory_mb"] = peak_memory_mb()
    try:
        import platform
        result["machine"] = {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__}
    except Exception:
        pass
    if args.run:
        Path(args.run).mkdir(parents=True, exist_ok=True)
        (Path(args.run) / "bench.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("synth", help="Make audio for test sets with one system")
    p.add_argument("--run", required=True)
    p.add_argument("--system", required=True, help="tiny-voca or kokoro")
    p.add_argument("--name", help="Name of the system in the run (default: --system)")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--voices", nargs="+")
    p.add_argument("--set", nargs="+", required=True, type=Path, help="Test set files (eval/sets/*.jsonl)")
    p.add_argument("--limit", type=int, help="First N sentences of each set (a quick trial)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=synth)

    p = commands.add_parser("refs", help="Add recordings as a system")
    p.add_argument("--run", required=True)
    p.add_argument("--system", required=True, help="Name, e.g. recordings or human")
    p.add_argument("--set", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path, help="Folder the set's audio paths are relative to")
    p.add_argument("--voice", default="human", help="Voice name for sets with one reference per sentence")
    p.add_argument("--limit", type=int)
    p.set_defaults(handler=refs)

    p = commands.add_parser("score", help="Score every clip in a run")
    p.add_argument("--run", required=True)
    p.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    p.add_argument("--system", nargs="+", help="Only these systems")
    p.add_argument("--asr-backend", choices=("mlx", "faster-whisper"))
    p.add_argument("--asr-model")
    p.add_argument("--device", default="mps" if apple_silicon() else "cpu")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=score)

    p = commands.add_parser("report", help="Summarise a run into report.json and report.md")
    p.add_argument("--run", required=True)
    p.add_argument("--reference", nargs="*", help="system:voice=refsystem:refvoice, for voice similarity")
    p.set_defaults(handler=report_command)

    p = commands.add_parser("pairs", help="Make a blind A/B listening test")
    p.add_argument("--run", required=True)
    p.add_argument("--a", required=True, help="system:voice")
    p.add_argument("--b", required=True, help="system:voice")
    p.add_argument("--set", nargs="*", help="Only sentences of these sets")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name")
    p.set_defaults(handler=pairs_command)

    p = commands.add_parser("bench", help="Speed and memory of a tiny-voca checkpoint")
    p.add_argument("--run")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--set", type=Path, default=SETS_DIR / "hard.jsonl")
    p.add_argument("--sentences", type=int, default=20)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--voice")
    p.set_defaults(handler=bench)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
