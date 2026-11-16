"""The training corpus builder on tiny fake copies of LibriTTS-R and Hi-Fi TTS."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import soundfile as sf
    import scipy  # noqa: F401
except ImportError:
    sf = None

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    """Imported by name, so worker processes can find its functions."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("build_corpus")


def tone(seconds, rate):
    time = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


@unittest.skipIf(sf is None, "soundfile and scipy are needed")
class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.sets = self.root / "sets"
        self.sets.mkdir()
        (self.sets / "hard.jsonl").write_text(json.dumps({"id": "h", "text": "This sentence is in the test set!"}) + "\n")
        self.build = load_script()

    def tearDown(self):
        self.folder.cleanup()

    def libritts(self):
        subset = self.root / "LibriTTS_R" / "train-clean-100"
        subset.mkdir(parents=True)
        (self.root / "LibriTTS_R" / "speakers.tsv").write_text("READER\tGENDER\tSUBSET\tNAME\n19\tF\ttrain-clean-100\tA\n"
                                                               "26\tM\ttrain-clean-100\tB\n")
        clips = {"19_198_000000_000000": ("Hello there, how are you today?", 3.0),
                 "19_198_000001_000000": ("this sentence is in the test set", 3.0),     # a test sentence: kept out
                 "19_198_000002_000000": ("Far too long to keep for training.", 25.0),
                 "19_198_000003_000000": ("Another fine sentence to keep.", 4.0),
                 "26_495_000000_000000": ("Only a little audio from this one.", 2.0)}  # speaker under the minimum
        for name, (text, seconds) in clips.items():
            folder = subset / name.split("_")[0] / name.split("_")[1]
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / f"{name}.wav", tone(seconds, 24000), 24000)
            (folder / f"{name}.normalized.txt").write_text(text + "\n")
        orphan = subset / "19" / "198" / "19_198_000009_000000.wav"
        sf.write(orphan, tone(2.0, 24000), 24000)                                     # no text: dropped
        return subset

    def hifitts(self):
        root = self.root / "hi_fi_tts_v0"
        lines = []
        for i in range(4):
            path = Path("audio") / "92_clean" / "10425" / f"clip_{i}.flac"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(root / path, tone(2.0, 44100), 44100)
            lines.append({"audio_filepath": str(path), "text": f"Clip {i}.", "text_normalized": f"Clip number {i} here.",
                          "duration": 2.0})
        (root / "92_manifest_clean_train.json").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (root / "92_manifest_other_train.json").write_text(json.dumps(lines[0]) + "\n")             # not used
        return root

    def run_build(self, *extra):
        output = self.root / "corpus"
        arguments = ["--output", str(output), "--libritts", str(self.libritts()), "--hifitts", str(self.hifitts()),
                     "--sets", str(self.sets), "--min-speaker-minutes", "0.1", "--min-validation-clips", "3",
                     "--validation-per-speaker", "1", "--workers", "2", *extra]
        return output, self.build.main(arguments)

    def rows(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_corpus(self):
        output, report = self.run_build()
        train, validation = self.rows(output / "train.jsonl"), self.rows(output / "validation.jsonl")
        ids = sorted(r["id"] for r in train + validation)
        self.assertEqual(len(ids), 6)
        self.assertIn("ltr-19_198_000000_000000", ids)
        self.assertNotIn("ltr-19_198_000001_000000", ids)
        self.assertNotIn("ltr-19_198_000002_000000", ids)
        self.assertNotIn("ltr-26_495_000000_000000", ids)
        self.assertEqual({r["voice_id"] for r in validation}, {"hifi-92"})                 # LibriTTS speaker has too few
        self.assertEqual(report["dropped"], {"no normalized text": 1, "sentence is in a test set": 1,
                                             "speaker has too little audio": 1, "too short or too long": 1})
        for row in train + validation:
            self.assertEqual(row["source_license"], "CC BY 4.0")
            self.assertTrue(Path(row["audio"]).is_absolute())
            info = sf.info(row["audio"])
            self.assertEqual((info.samplerate, info.channels), (24000, 1))
        hifi = next(r for r in train + validation if r["voice_id"] == "hifi-92")
        self.assertTrue(hifi["text"].startswith("Clip number"))
        self.assertAlmostEqual(hifi["seconds"], 2.0, places=2)
        speakers = json.loads((output / "speakers.json").read_text())
        self.assertEqual([s["id"] for s in speakers], ["hifi-92", "ltr-19"])
        self.assertEqual(speakers[1]["gender"], "F")

    def test_rerun_gives_the_same_files(self):
        output, _ = self.run_build()
        first = {name: (output / name).read_bytes() for name in ("train.jsonl", "validation.jsonl", "speakers.json")}
        self.build.main(["--output", str(output), "--libritts", str(self.root / "LibriTTS_R" / "train-clean-100"),
                         "--hifitts", str(self.root / "hi_fi_tts_v0"), "--sets", str(self.sets), "--min-speaker-minutes", "0.1",
                         "--min-validation-clips", "3", "--validation-per-speaker", "1", "--workers", "2"])
        for name, content in first.items():
            self.assertEqual((output / name).read_bytes(), content, name)

    def test_prepare_accepts_the_rows(self):
        output, _ = self.run_build()
        from tiny_voca.config import Config
        speakers = [s["id"] for s in json.loads((output / "speakers.json").read_text())]
        config = Config(speakers=tuple(speakers))
        for row in self.rows(output / "train.jsonl"):
            self.assertIn(row["voice_id"], config.speakers)
            wave, rate = sf.read(Path(output) / row["audio"], dtype="float32")              # absolute path wins
            self.assertEqual(rate, config.sample_rate)
            self.assertEqual(row["split"], "train")


if __name__ == "__main__":
    unittest.main()
