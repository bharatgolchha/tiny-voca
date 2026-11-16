"""The scoreboard's bookkeeping: test sets, synthesis rows, the report and the
A/B pairs. The scoring models themselves (UTMOSv2, Whisper, WavLM) are not
loaded here; their outputs are written as they would be."""

import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import torch

from tiny_voca import scoreboard
from tiny_voca.model import TinyVoca
from tiny_voca.text import tokenize
from tiny_voca.train import export_model
from helpers import small_config

SETS = Path(__file__).resolve().parents[1] / "eval" / "sets"


def phonemize(text):
    pieces = []
    for word, mark in re.findall(r"([A-Za-z0-9']+)|([,.?!;:])", text):
        pieces.append((True, mark) if mark else (False, [[("t", 0), ("ɑː", 1), ("n", 0)]]))
    return tokenize(pieces)


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class HardSetTest(unittest.TestCase):
    def test_hard_set_is_frozen_and_clean(self):
        rows = scoreboard.load_set(SETS / "hard.jsonl")
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({r["text"] for r in rows}), 200)
        self.assertEqual(len({r["category"] for r in rows}), 12)
        for row in rows:
            self.assertTrue(row["id"].startswith("hard-" + row["category"]))
            self.assertEqual(row["text"], row["text"].strip())

    def test_load_set_rejects_duplicates_and_empty_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.jsonl"
            write(path, [{"id": "a", "text": "One."}, {"id": "a", "text": "Two."}])
            with self.assertRaises(ValueError):
                scoreboard.load_set(path)
            write(path, [{"id": "a", "text": " "}])
            with self.assertRaises(ValueError):
                scoreboard.load_set(path)
            write(path, [{"id": "a", "text": "One."}, {"id": "b", "text": "Two."}])
            self.assertEqual([r["id"] for r in scoreboard.load_set(path, limit=1)], ["a"])


class StatisticsTest(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(scoreboard.sign_test(23, 7), 0.0026, places=4)
        self.assertAlmostEqual(scoreboard.sign_test(15, 15), 0.5722, places=4)
        self.assertEqual(scoreboard.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(scoreboard.sign_test(3, 0), 0.125)

    def test_mean_ci(self):
        mean, half = scoreboard.mean_ci([3.0, 4.0, 5.0])
        self.assertAlmostEqual(mean, 4.0)
        self.assertAlmostEqual(half, 1.96 * 1.0 / math.sqrt(3))
        self.assertEqual(scoreboard.mean_ci([2.0]), (2.0, None))
        self.assertEqual(scoreboard.mean_ci([]), (None, None))

    def test_consistency_is_leave_one_out(self):
        same = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        self.assertAlmostEqual(scoreboard.consistency(same), 1.0)
        spread = [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(scoreboard.consistency(spread), 0.0)
        self.assertIsNone(scoreboard.consistency([[1.0, 0.0]]))

    def test_word_errors(self):
        try:
            import jiwer  # noqa: F401
        except ImportError:
            self.skipTest("jiwer is not installed")
        self.assertEqual(scoreboard.word_errors("the cat sat", "the cat sat"), (0, 3))
        self.assertEqual(scoreboard.word_errors("the cat sat", "the dog sat down"), (2, 3))
        self.assertEqual(scoreboard.word_errors("the cat", ""), (2, 2))


class RunTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.run = self.root / "run"

    def tearDown(self):
        self.folder.cleanup()

    def fake_run(self):
        """Two systems, one voice each, three sentences, with scores as the
        scoring step writes them."""
        clips = []
        for system, voice, seconds in (("new", "A", 0.1), ("old", "B", None)):
            for i in range(3):
                path = scoreboard.audio_path(self.run, system, voice, "hard", f"s{i}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
                clips.append({"key": scoreboard.clip_key(system, voice, "hard", f"s{i}"), "system": system, "voice": voice,
                              "set": "hard", "id": f"s{i}", "text": f"sentence {i}", "category": "numbers" if i else "names",
                              "path": str(path.relative_to(self.run)), "audio_seconds": 2.0, "synth_seconds": seconds})
        scoreboard.append_jsonl(self.run / "clips.jsonl", clips)
        scoreboard.append_jsonl(self.run / "scores-utmos.jsonl", [{"key": c["key"], "utmos": 4.0 if c["system"] == "new" else 3.0}
                                                                  for c in clips])
        scoreboard.append_jsonl(self.run / "scores-asr.jsonl", [{"key": c["key"], "asr_model": "test", "transcript": "",
                                                                 "errors": 0 if c["system"] == "new" else 1, "words": 4} for c in clips])
        scoreboard.append_jsonl(self.run / "scores-voice.jsonl", [{"key": c["key"], "embedding": [1.0, 0.0] if c["system"] == "new" else [0.6, 0.8]}
                                                                  for c in clips])
        return clips

    def test_report(self):
        self.fake_run()
        report = scoreboard.build_report(self.run, ["new:A=old:B"])
        rows = {r["system"]: r for r in report["rows"]}
        self.assertEqual(rows["new"]["utmos"], 4.0)
        self.assertEqual(rows["old"]["wer"], 0.25)
        self.assertEqual(rows["new"]["wer"], 0.0)
        self.assertEqual(rows["old"]["wer_by_category"], {"names": 0.25, "numbers": 0.25})
        self.assertEqual(rows["new"]["times_real_time"], 20.0)
        self.assertNotIn("times_real_time", rows["old"])
        self.assertAlmostEqual(rows["new"]["voice_similarity"], 0.6)
        self.assertAlmostEqual(rows["new"]["voice_consistency"], 1.0)
        self.assertNotIn("voice_similarity", rows["old"])
        text = scoreboard.report_markdown(report)
        self.assertIn("| new | A | hard | 3 | 4.00 ± 0.00 | 0.0% | 0.600 | 1.000 | 20 |", text)

    def test_resynthesis_replaces_the_row(self):
        clips = self.fake_run()
        scoreboard.append_jsonl(self.run / "clips.jsonl", [{**clips[0], "audio_seconds": 9.0}])
        self.assertEqual(scoreboard.load_clips(self.run)[clips[0]["key"]]["audio_seconds"], 9.0)
        self.assertEqual(len(scoreboard.load_clips(self.run)), 6)

    def test_pairs_are_blind_balanced_and_repeatable(self):
        clips = scoreboard.load_clips(self.run) if self.fake_run() else None
        test = scoreboard.make_pairs(clips, "new:A", "old:B", None, 30, seed=4)
        self.assertEqual(len(test["pairs"]), 3)
        again = scoreboard.make_pairs(clips, "new:A", "old:B", None, 30, seed=4)
        self.assertEqual(test, again)
        for pair in test["pairs"]:
            self.assertNotEqual(pair["first"], pair["second"])
            first_system = pair["first"].split("/")[2]
            self.assertEqual(first_system, "new" if pair["first_is"] == "a" else "old")
            self.assertTrue(pair["first"].startswith("../audio/"))
        self.assertIn(sum(p["first_is"] == "a" for p in test["pairs"]), (1, 2))
        with self.assertRaises(SystemExit):
            scoreboard.make_pairs(clips, "new:A", "nobody:X", None, 30, seed=4)

    def test_refs_copies_recordings_into_the_run(self):
        import numpy as np
        import soundfile as sf
        data = self.root / "data"
        (data / "v").mkdir(parents=True)
        for voice in ("F2", "M2"):
            sf.write(data / "v" / f"{voice}.wav", np.zeros(2400, dtype=np.float32), 24000)
        sets = self.root / "heldout.jsonl"
        write(sets, [{"id": "t1", "text": "Hello.", "refs": {"F2": "v/F2.wav", "M2": "v/M2.wav"}}])
        scoreboard.main(["refs", "--run", str(self.run), "--system", "recordings", "--set", str(sets), "--data-root", str(data)])
        clips = scoreboard.load_clips(self.run)
        self.assertEqual(sorted(clips), ["recordings/F2/heldout/t1", "recordings/M2/heldout/t1"])
        self.assertAlmostEqual(clips["recordings/F2/heldout/t1"]["audio_seconds"], 0.1)
        self.assertTrue((self.run / clips["recordings/F2/heldout/t1"]["path"]).exists())
        scoreboard.main(["refs", "--run", str(self.run), "--system", "recordings", "--set", str(sets), "--data-root", str(data)])
        self.assertEqual(len(scoreboard.read_jsonl(self.run / "clips.jsonl")), 2)      # nothing added twice

    def test_synth_with_a_tiny_voca_checkpoint(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile is not installed")
        torch.manual_seed(0)
        checkpoint = self.root / "model.pt"
        export_model(TinyVoca(small_config()), checkpoint, step=1)
        sets = self.root / "mini.jsonl"
        write(sets, [{"id": "a", "text": "Hello there.", "category": "greeting"}, {"id": "b", "text": "Two words"},
                     {"id": "c", "text": "!!!"}])
        with patch("tiny_voca.infer.phonemize", phonemize), patch("tiny_voca.scoreboard.os.environ", {"TINY_VOCA_FRONTEND": "test"}):
            arguments = ["synth", "--run", str(self.run), "--system", "tiny-voca", "--checkpoint", str(checkpoint),
                         "--voices", "F2", "--set", str(sets), "--threads", "1"]
            scoreboard.main(arguments)
            scoreboard.main(arguments)                      # resumes: nothing is made twice
        clips = scoreboard.load_clips(self.run)
        self.assertEqual(sorted(clips), ["tiny-voca/F2/mini/a", "tiny-voca/F2/mini/b"])
        failures = scoreboard.read_jsonl(self.run / "failures.jsonl")
        self.assertEqual([f["key"] for f in failures], ["tiny-voca/F2/mini/c"] * 2)       # tried again on resume
        self.assertIn("NothingToPronounce", failures[0]["error"])
        self.assertEqual([f["key"] for f in scoreboard.build_report(self.run)["failures"]], ["tiny-voca/F2/mini/c"] * 2)
        self.assertEqual(len(scoreboard.read_jsonl(self.run / "clips.jsonl")), 2)
        clip = clips["tiny-voca/F2/mini/a"]
        self.assertEqual(clip["category"], "greeting")
        self.assertGreater(clip["audio_seconds"], 0)
        self.assertGreaterEqual(clip["synth_seconds"], 0)
        import soundfile as sf
        info = sf.info(self.run / clip["path"])
        self.assertEqual(info.samplerate, 24000)


if __name__ == "__main__":
    unittest.main()
