"""Train waveform heads alone, on recorded features, to compare them fairly.

    tiny-voca-vocoder --features /workspace/data/corpus/features --config /workspace/data/corpus/config.json \\
        --output /workspace/runs/vocoder --heads source_filter neural --device cuda --max-seconds 10800

Each head named in ``--heads`` gets its own conditioning network (recorded mel
spectrum, pitch and voicing in, the model's hidden width out), its own critics
and optimisers, and every head trains on the very same batches in one process.
A comparison therefore differs only in the head. The oscillator follows the
recorded pitch, exactly as in full-model training. Only the feature files are
read, so no alignment is needed.

Writes, in ``--output``:

* ``<head>/last.pt``: everything needed to resume (``--resume``)
* ``<head>/vocoder.pt``: configuration and generator weights
* ``<head>/metrics.jsonl``: training and validation losses
* ``samples/<head>/<n>.wav`` and ``samples/recording/<n>.wav``: the same
  validation sentences rendered by each head, next to the recording, written
  at the end (or with ``--render-only``) for a blind listening test

Running this starts paid compute when ``--device`` is a rented GPU.
"""

import argparse
from collections import defaultdict
import dataclasses
import json
import math
from pathlib import Path
import time

import soundfile as sf
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import Config
from .data import continuous_pitch, float_wave
from .losses import Discriminators, SpectralLoss, discriminator_loss, generator_gan_loss
from .model import ConvNeXtBlock, NeuralFilter, SourceFilter, length_mask
from .train import atomic_save, learning_rate

HEADS = ("source_filter", "neural")
FORMAT = 1


class Conditioner(nn.Module):
    """Recorded mel spectrum, log pitch and voicing to the hidden state a head reads.

    It stands in for the acoustic stack of the full model, so both heads get the
    same, reasonably strong, input. Training-only: the full model brings its own.
    """

    def __init__(self, c, layers=4):
        super().__init__()
        self.config = c
        self.input = nn.Linear(c.n_mels + 2, c.channels)
        self.blocks = nn.ModuleList(ConvNeXtBlock(c.channels, 3, 7, 1.0 / layers) for _ in range(layers))
        self.norm = nn.LayerNorm(c.channels)

    def forward(self, mel, pitch, voiced, mask):
        c = self.config
        x = torch.cat(((mel - c.mel_mean) / c.mel_scale, pitch[..., None], voiced[..., None]), dim=-1)
        x = self.input(x) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) * mask[..., None]


class Vocoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.conditioner = Conditioner(c)
        self.head = NeuralFilter(c) if c.vocoder == "neural" else SourceFilter(c)

    def forward(self, batch, noise=None, phase=None):
        c = self.config
        mask = length_mask(batch["lengths"], batch["mel"].shape[1]).float()
        hidden = self.conditioner(batch["mel"], batch["pitch"], batch["voiced"], mask)
        f0 = (c.pitch_reference_hz * torch.exp(batch["pitch"].float())).clamp(c.f0_min, c.f0_max)
        return self.head(hidden, f0, batch["lengths"], noise=noise, phase=phase)


def read_rows(manifest, config):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").split("\n") if line.strip()]
    if any(row.get("acoustic") != config.acoustic_fingerprint for row in rows):
        raise ValueError(f"{manifest}: acoustic feature settings differ from this configuration")
    return rows


def load_features(root, row, c):
    root = Path(root)
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("feature file escapes the feature folder")
    stored = torch.load(path, map_location="cpu", weights_only=True)
    frames = row["frames"]
    if stored["mel"].shape != (frames, c.n_mels) or stored["prosody"].shape != (frames, 3) or stored["wave"].shape != (frames * c.hop,):
        raise ValueError(f"feature dimensions differ from the manifest: {path.name}")
    pitch, _ = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
    return {"mel": stored["mel"].float(), "pitch": pitch.float(), "voiced": stored["prosody"][:, 1].float(),
            "wave": float_wave(stored["wave"])}


class Crops(torch.utils.data.Dataset):
    """One random window of ``frames`` frames from each clip at least that long."""

    def __init__(self, manifest, config, frames):
        self.root, self.config, self.frames = Path(manifest).parent, config, frames
        self.rows = [r for r in read_rows(manifest, config) if r["frames"] >= frames]
        if not self.rows:
            raise ValueError(f"no clip in {manifest} has {frames} frames")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = load_features(self.root, self.rows[index], self.config)
        start = int(torch.randint(self.rows[index]["frames"] - self.frames + 1, ()))
        hop = self.config.hop
        return {"mel": item["mel"][start:start + self.frames], "pitch": item["pitch"][start:start + self.frames],
                "voiced": item["voiced"][start:start + self.frames],
                "wave": item["wave"][start * hop:(start + self.frames) * hop]}


def collate(items):
    batch = {key: pad_sequence([x[key] for x in items], batch_first=True) for key in ("mel", "pitch", "voiced", "wave")}
    batch["lengths"] = torch.tensor([x["mel"].shape[0] for x in items])
    return batch


def to(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batches(dataset, size, workers, seed):
    """Endless shuffled batches; each pass over the clips uses a new order."""
    generator = torch.Generator().manual_seed(seed)
    while True:
        loader = torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=True, drop_last=True, num_workers=workers,
                                             collate_fn=collate, generator=generator, pin_memory=torch.cuda.is_available(),
                                             persistent_workers=False, prefetch_factor=4 if workers else None)
        yield from loader


def validation_rows(rows, count):
    """A fixed, speaker-balanced choice of held-out clips, shortest speakers' order stable."""
    by_speaker = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        by_speaker[row["speaker"]].append(row)
    chosen, offset = [], 0
    while len(chosen) < min(count, len(rows)):
        for speaker in sorted(by_speaker):
            if offset < len(by_speaker[speaker]) and len(chosen) < count:
                chosen.append(by_speaker[speaker][offset])
        offset += 1
    return chosen


class Head:
    """One head under training: generator, critics, their optimisers and log."""

    def __init__(self, name, config, recipe, folder, device):
        self.name, self.folder, self.recipe = name, Path(folder), recipe
        self.config = dataclasses.replace(config, vocoder=name)
        self.generator = Vocoder(self.config).to(device)
        self.critics = Discriminators().to(device)
        self.spectral = SpectralLoss(self.config.sample_rate).to(device)
        self.optimizer = torch.optim.AdamW(self.generator.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.01)
        self.critic_optimizer = torch.optim.AdamW(self.critics.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.0)
        self.window = defaultdict(float)

    def parameters(self):
        return {"head": sum(p.numel() for p in self.generator.head.parameters()),
                "conditioner": sum(p.numel() for p in self.generator.conditioner.parameters()),
                "critics": sum(p.numel() for p in self.critics.parameters())}

    def step(self, batch, step):
        recipe, weights = self.recipe, self.recipe["weights"]
        rate = learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"])
        for group in self.optimizer.param_groups + self.critic_optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        phase = 2 * math.pi * torch.rand(len(batch["lengths"]), device=batch["wave"].device)
        wave = self.generator(batch, phase=phase)
        stft, mel = self.spectral(wave, batch["wave"], batch["lengths"] * self.config.hop)
        reconstruction = weights["stft"] * stft + weights["mel"] * mel
        loss, metrics = reconstruction, {"stft": stft, "mel": mel}
        if step >= recipe["gan_start"]:
            real = batch["wave"]
            self.critic_optimizer.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(self.critics(real), self.critics(wave.detach()))
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(), 10, error_if_nonfinite=True)
            self.critic_optimizer.step()
            self.critics.requires_grad_(False)
            try:
                with torch.no_grad():
                    real_features = self.critics(real)
                adversarial, matching = generator_gan_loss(real_features, self.critics(wave))
                gan = adversarial + recipe["feature_matching"] * matching
                # The same balance as full-model training: the adversarial gradient at
                # the waveform is gan_ratio times the reconstruction gradient.
                reconstruction_gradient = torch.autograd.grad(reconstruction, wave, retain_graph=True)[0]
                gan_gradient = torch.autograd.grad(gan, wave, retain_graph=True)[0]
                ramp = min(1.0, (step - recipe["gan_start"] + 1) / max(recipe["gan_ramp"], 1))
                scale = recipe["gan_ratio"] * ramp * reconstruction_gradient.norm() / gan_gradient.norm().clamp_min(1e-12)
                scale = scale.clamp(max=recipe["gan_scale_max"]).detach()
                loss = loss + scale * gan
                metrics.update(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, gan_scale=scale)
                loss.backward()
            finally:
                self.critics.requires_grad_(True)
        else:
            loss.backward()
        grad = nn.utils.clip_grad_norm_(self.generator.parameters(), recipe["grad_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        for key, value in {**metrics, "loss": loss, "grad_norm": grad}.items():
            self.window[key] += float(value.detach())

    @torch.no_grad()
    def validate(self, items, device, batch_size=4):
        """Spectral distances on whole held-out clips, with fixed noise and phase."""
        self.generator.eval()
        totals = defaultdict(float)
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = to(collate(chunk), device)
            noise = torch.randn(batch["wave"].shape, generator=torch.Generator().manual_seed(start)).to(device)
            wave = self.generator(batch, noise=noise)
            stft, mel = self.spectral(wave, batch["wave"], batch["lengths"] * self.config.hop)
            totals["validation_stft"] += float(stft) * len(chunk)
            totals["validation_mel"] += float(mel) * len(chunk)
        self.generator.train()
        return {key: value / len(items) for key, value in totals.items()}

    def state(self, step):
        return {"format": FORMAT, "config": self.config.to_dict(), "step": step, "recipe": self.recipe,
                "generator": self.generator.state_dict(), "critics": self.critics.state_dict(),
                "optimizer": self.optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict()}

    def save(self, step):
        self.folder.mkdir(parents=True, exist_ok=True)
        atomic_save(self.state(step), self.folder / "last.pt")
        atomic_save({"format": FORMAT, "config": self.config.to_dict(), "step": step,
                     "generator": {k: v.detach().cpu() for k, v in self.generator.state_dict().items()}}, self.folder / "vocoder.pt")

    def load(self, path, device, recipe_must_match=True):
        state = torch.load(path, map_location=device, weights_only=True)
        if state.get("format") != FORMAT or state["config"] != self.config.to_dict():
            raise ValueError(f"{path} was trained with a different configuration")
        if recipe_must_match and state["recipe"] != self.recipe:
            raise ValueError(f"{path} was trained with a different recipe")
        self.generator.load_state_dict(state["generator"])
        if "critics" in state:
            self.critics.load_state_dict(state["critics"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        return state["step"]

    def log(self, record):
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


@torch.no_grad()
def render(heads, items, rows, folder, device):
    """The same held-out clips from every head, and the recordings, as 16-bit wav."""
    folder = Path(folder)
    for head in heads:
        head.generator.eval()
    (folder / "recording").mkdir(parents=True, exist_ok=True)
    listing = []
    for n, (item, row) in enumerate(zip(items, rows), 1):
        name = f"{n:03d}.wav"
        sf.write(folder / "recording" / name, item["wave"].numpy(), heads[0].config.sample_rate, subtype="PCM_16")
        batch = to(collate([item]), device)
        noise = torch.randn(batch["wave"].shape, generator=torch.Generator().manual_seed(n)).to(device)
        for head in heads:
            (folder / head.name).mkdir(parents=True, exist_ok=True)
            wave = head.generator(batch, noise=noise)[0].clamp(-1, 1).cpu().numpy()
            sf.write(folder / head.name / name, wave, head.config.sample_rate, subtype="PCM_16")
        listing.append({"file": name, "id": row["id"], "speaker": row["speaker"], "text": row.get("text", "")})
    (folder / "clips.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for head in heads:
        head.generator.train()


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="Feature folder with train.jsonl and validation.jsonl")
    parser.add_argument("--config", type=Path, help="Corpus config.json (the acoustic settings must match the features)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-time cap; save, render and exit at a step boundary")
    parser.add_argument("--stop-file", type=Path, help="Save, render and exit after a step when this file appears")
    parser.add_argument("--crop-frames", type=int, default=64, help="Frames per training window (64 = 0.68 s)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="Processes reading feature files")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr-half-life", type=float, default=100000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--stft-weight", type=float, default=2.0)
    parser.add_argument("--mel-weight", type=float, default=10.0)
    parser.add_argument("--gan-start", type=int, default=2000)
    parser.add_argument("--gan-ramp", type=int, default=2000)
    parser.add_argument("--gan-ratio", type=float, default=0.7)
    parser.add_argument("--feature-matching", type=float, default=30.0)
    parser.add_argument("--gan-scale-max", type=float, default=2000.0)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--render", type=int, default=40, help="Held-out clips rendered by every head at the end")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--resume", action="store_true", help="Continue the run in --output (the data order starts afresh)")
    parser.add_argument("--render-only", action="store_true", help="Render samples from the saved heads in --output and exit")
    args = parser.parse_args(arguments)
    if min(args.steps, args.crop_frames, args.batch_size, args.save_every, args.log_every, args.validation_size) < 1 or args.workers < 0:
        parser.error("steps, sizes and intervals must be positive")
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    recipe = {key: getattr(args, key) for key in ("crop_frames", "batch_size", "learning_rate", "warmup", "lr_half_life", "grad_clip",
                                                   "gan_start", "gan_ramp", "gan_ratio", "feature_matching", "gan_scale_max", "seed")}
    recipe["weights"] = {"stft": args.stft_weight, "mel": args.mel_weight}
    heads = [Head(name, c, recipe, args.output / name, args.device) for name in args.heads]
    validation = read_rows(args.features / "validation.jsonl", c)
    held_out = validation_rows(validation, max(args.validation_size, args.render))
    items = [load_features(args.features, row, c) for row in held_out]
    if args.render_only:
        for head in heads:
            head.load(head.folder / "vocoder.pt", args.device, recipe_must_match=False)
        render(heads, items[:args.render], held_out[:args.render], args.output / "samples", args.device)
        print(json.dumps({"rendered": min(args.render, len(items)), "samples": str(args.output / "samples")}))
        return
    step = 0
    if args.resume:
        steps = {head.load(head.folder / "last.pt", args.device) for head in heads}
        if len(steps) != 1:
            raise ValueError("the heads in this run were saved at different steps")
        step = steps.pop()
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recipe.json").write_text(json.dumps({**recipe, "heads": args.heads, "config": c.to_dict()}, indent=1) + "\n")
    data = Crops(args.features / "train.jsonl", c, args.crop_frames)
    stream = batches(data, args.batch_size, args.workers, args.seed + step)
    print(json.dumps({"train_clips": len(data), "validation_clips": len(items), "start_step": step,
                      "parameters": {head.name: head.parameters() for head in heads}}), flush=True)
    start = window_start = time.monotonic()
    window_steps, waited = 0, 0.0
    while step < args.steps:
        ready = time.monotonic()
        batch = to(next(stream), args.device)
        waited += time.monotonic() - ready
        for head in heads:
            head.step(batch, step)
        step += 1
        window_steps += 1
        elapsed = time.monotonic() - start
        stop = elapsed >= args.max_seconds or step == args.steps or (args.stop_file is not None and args.stop_file.exists())
        save = step % args.save_every == 0 or stop
        if step % args.log_every == 0 or save:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = max(time.monotonic() - window_start, 1e-9)
            for head in heads:
                record = {key: value / window_steps for key, value in head.window.items()}
                record.update(step=step, elapsed_seconds=round(elapsed, 1), steps_per_second=window_steps / seconds,
                              data_wait_fraction=waited / seconds, learning_rate=head.optimizer.param_groups[0]["lr"])
                if save:
                    record.update(head.validate(items[:args.validation_size], args.device))
                    head.save(step)
                head.log(record)
                print(json.dumps({"head": head.name, **record}), flush=True)
                head.window = defaultdict(float)
            window_start, window_steps, waited = time.monotonic(), 0, 0.0
        if stop:
            break
    render(heads, items[:args.render], held_out[:args.render], args.output / "samples", args.device)
    print(json.dumps({"done": step, "samples": str(args.output / "samples")}), flush=True)


if __name__ == "__main__":
    main()
