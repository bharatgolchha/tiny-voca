import base64
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

import soundfile as sf
import torch

from helpers import small_config
from tiny_voca import vocoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ab_page  # noqa: E402


def write_features(root, c, counts=(("train", 6), ("validation", 4))):
    """Feature files and manifests shaped like tiny-voca-prepare's."""
    root = Path(root)
    for split, count in counts:
        (root / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(count):
            frames = 40 + 7 * i
            time = torch.arange(frames * c.hop) / c.sample_rate
            hz = 110 + 40 * i
            voiced = torch.ones(frames)
            voiced[:3] = 0
            prosody = torch.stack((math.log(hz / c.pitch_reference_hz) * voiced, voiced, torch.full((frames,), -3.0)), dim=-1)
            wave = (0.2 * torch.sin(2 * math.pi * hz * time) * 32767).to(torch.int16)
            path = f"{split}/{split}{i}.pt"
            torch.save({"wave": wave, "mel": torch.randn(frames, c.n_mels) - 5, "prosody": prosody}, root / path)
            rows.append({"id": f"{split}{i}", "path": path, "frames": frames, "acoustic": c.acoustic_fingerprint,
                         "speaker": c.speakers[i % 2], "text": f"sentence {i}"})
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {**c.to_dict(), "head_channels": 16, "head_layers": 2}
    (root / "config.json").write_text(json.dumps(config))
    return root


class VocoderTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.c = small_config()
        self.features = write_features(self.root / "features", self.c)

    def tearDown(self):
        self.folder.cleanup()

    def arguments(self, *extra):
        return ["--features", str(self.features), "--config", str(self.features / "config.json"), "--output", str(self.root / "run"),
                "--device", "cpu", "--workers", "0", "--batch-size", "2", "--crop-frames", "32", "--gan-start", "2", "--gan-ramp", "2",
                "--save-every", "3", "--log-every", "1", "--validation-size", "2", "--render", "3", *extra]

    def test_crops_keep_audio_and_features_aligned(self):
        c = self.c
        folder = self.root / "aligned"
        (folder / "train").mkdir(parents=True)
        frames = 50
        torch.save({"wave": torch.arange(frames).repeat_interleave(c.hop).float() / 1000, "mel": torch.arange(frames).float()[:, None].repeat(1, c.n_mels),
                    "prosody": torch.zeros(frames, 3)}, folder / "train" / "a.pt")
        (folder / "train.jsonl").write_text(json.dumps({"id": "a", "path": "train/a.pt", "frames": frames, "acoustic": c.acoustic_fingerprint,
                                                        "speaker": c.speakers[0]}) + "\n")
        data = vocoder.Crops(folder / "train.jsonl", c, 16)
        for _ in range(10):
            item = data[0]
            self.assertEqual(item["mel"].shape, (16, c.n_mels))
            self.assertEqual(item["wave"].shape, (16 * c.hop,))
            self.assertAlmostEqual(float(item["wave"][0]) * 1000, float(item["mel"][0, 0]), places=3)
            self.assertAlmostEqual(float(item["wave"][-1]) * 1000, float(item["mel"][-1, 0]), places=3)

    def test_short_clips_and_other_features_are_refused(self):
        with self.assertRaises(ValueError):
            vocoder.Crops(self.features / "train.jsonl", self.c, 10_000)
        other = type(self.c).from_dict({**self.c.to_dict(), "n_fft": 2048})
        with self.assertRaises(ValueError):
            vocoder.Crops(self.features / "train.jsonl", other, 32)

    def test_both_heads_train_on_the_same_batches_save_resume_and_render(self):
        vocoder.main(self.arguments("--steps", "4"))
        run = self.root / "run"
        for head in vocoder.HEADS:
            records = [json.loads(line) for line in (run / head / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in records], [1, 2, 3, 4])
            self.assertIn("discriminator", records[-1])           # the adversarial phase ran
            self.assertIn("validation_mel", records[2])
            self.assertTrue(all(math.isfinite(r["loss"]) for r in records))
            saved = torch.load(run / head / "vocoder.pt", weights_only=True)
            self.assertEqual((saved["step"], saved["config"]["vocoder"]), (4, head))
        clips = json.loads((run / "samples" / "clips.json").read_text())
        self.assertEqual(len(clips), 3)
        self.assertEqual(len({c["speaker"] for c in clips}), 2)   # balanced over speakers
        for folder in ("recording", *vocoder.HEADS):
            wave, rate = sf.read(run / "samples" / folder / "001.wav")
            self.assertEqual(rate, self.c.sample_rate)
            self.assertGreater(len(wave), 0)
        with self.assertRaises(FileExistsError):
            vocoder.main(self.arguments("--steps", "6"))
        vocoder.main(self.arguments("--steps", "6", "--resume"))
        records = [json.loads(line) for line in (run / "neural" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(records[-1]["step"], 6)
        with self.assertRaises(ValueError):                   # a changed recipe is a new run, not a resume
            vocoder.main(self.arguments("--steps", "8", "--resume", "--learning-rate", "1e-3"))
        before = (run / "samples" / "neural" / "002.wav").read_bytes()
        vocoder.main(self.arguments("--render-only"))
        self.assertEqual((run / "samples" / "neural" / "002.wav").read_bytes(), before)   # fixed noise: rendering repeats
        ab_page.main(["--samples", str(run / "samples")])
        text = (run / "samples" / "ab.html").read_text()
        key = json.loads(base64.b64decode(re.search(r"atob\('([^']+)'\)", text).group(1)))
        self.assertEqual([sorted(pair) for pair in key], [["neural", "source_filter"]] * 3)
        self.assertNotIn("neural", text.split("<script>")[0])        # file names do not give the heads away
        self.assertTrue((run / "samples" / "ab" / "001-a.wav").is_file())

    def test_heads_differ_only_in_the_head(self):
        heads = [vocoder.Head(name, type(self.c).from_dict(json.loads((self.features / "config.json").read_text())),
                              {"learning_rate": 1e-3}, self.root / name, "cpu") for name in vocoder.HEADS]
        conditioners = [h.parameters()["conditioner"] for h in heads]
        self.assertEqual(conditioners[0], conditioners[1])
        self.assertEqual(heads[0].parameters()["critics"], heads[1].parameters()["critics"])
        self.assertNotEqual(heads[0].parameters()["head"], heads[1].parameters()["head"])


if __name__ == "__main__":
    unittest.main()
