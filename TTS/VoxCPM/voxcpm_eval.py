#!/usr/bin/env python3
"""
voxcpm_eval.py — VoxCPM2 cloning quality + speed harness (Titan V, isolated).

For each input sentence, synthesizes the cloned voice in TWO modes and
compares them:

  * "ultimate"     — reference wav + its transcript (audio-continuation cloning)
  * "controllable" — reference wav only (no transcript)

For each (sentence, mode) it records:
  * RTF            — generate wall time / output audio duration (lower = faster)
  * first_chunk_s  — latency to the first streamed audio chunk (streaming mode)
  * sim            — ECAPA-TDNN cosine speaker similarity vs the reference clip
                     (higher = closer to the target voice; ~0.7+ is "same speaker")

Intelligibility (WER) is intentionally left out — check that by ear.

Inputs (default layout, override via flags):
  tests/Alice_VoiceSample.wav     reference voice (10s clone source)
  tests/Alice_VoiceSample.txt     transcript of the reference (for "ultimate")
  --sentences file (one line per sentence) OR the built-in defaults below

Outputs:
  out/<mode>/<NN>.wav             each synthesized sentence
  out/results.csv                 the metrics table
  out/streaming/<mode>_<NN>.wav   streamed renders (when --streaming)

Usage:
  python voxcpm_eval.py \
      --ref tests/Alice_VoiceSample.wav \
      --ref-text tests/Alice_VoiceSample.txt \
      --model openbmb/VoxCPM2 \
      --device cuda \
      --streaming

Notes:
  * VoxCPM2 wants CUDA >= 12.0; on the Titan V (sm_70) it should run but is the
    trailing edge. If CUDA/kernel errors appear, that's the old card, not the
    harness — try --device cpu to confirm the pipeline, or use the A6000.
  * The exact streaming clone-arg signature isn't fully documented; this script
    passes the same clone kwargs to generate_streaming() and FAILS LOUDLY with a
    hint if that call shape is wrong, so you can adjust one place (STREAM_KWARGS).
"""

import argparse
import csv
import os
import sys
import time
import wave
import contextlib

import numpy as np

# ---------------------------------------------------------------------------
# torch.compile / TorchDynamo control.
#
# On Volta (Titan V) the compile path is unreliable (bf16 unsupported, einops
# tracing errors). We let the user disable it. This must happen BEFORE torch is
# imported by anything heavy, so we set the env var here based on argv/env.
#
#   --no-compile  -> TORCHDYNAMO_DISABLE=1 (global, bluntest, most reliable)
#   --compile     -> leave compilation on (Ampere+; lets CUDA graphs fuse the
#                    many small AR kernel launches — usually the big speedup)
# Default: no-compile (safe everywhere; flip on for the A6000 matrix).
# ---------------------------------------------------------------------------
def _early_compile_setup():
    argv = sys.argv
    want_compile = ("--compile" in argv)
    want_nocompile = ("--no-compile" in argv)
    # env override if no explicit flag
    if not want_compile and not want_nocompile:
        env = os.environ.get("VOXCPM_COMPILE", "0")
        want_compile = env not in ("0", "", "false", "False")
    if not want_compile:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # Always make dynamo fall back to eager instead of hard-failing, as a belt-
    # and-suspenders guard even when compile is requested.
    try:
        import torch._dynamo  # noqa
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass


_early_compile_setup()


# ---- default eval sentences (override with --sentences) ---------------------
DEFAULT_SENTENCES = [
    "The quarterly report shows a twelve percent increase in net revenue.",
    "Please configure the Kubernetes ingress before deploying the service.",
    "She said the appointment was rescheduled to next Thursday afternoon.",
    "Our latency dropped below two hundred milliseconds after the refactor.",
    "Can you summarize the key findings in three short bullet points?",
]


def log(msg):
    print(f"[voxcpm-eval] {msg}", flush=True)


def log_device_info(device, dtype, compile_on):
    """Log torch/CUDA/card info so each run records what it actually ran at."""
    try:
        import torch
        log(f"torch {torch.__version__}  cuda_build={torch.version.cuda}")
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = 0 if ":" not in device else int(device.split(":")[1])
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            bf16_ok = torch.cuda.is_bf16_supported()
            log(f"GPU: {name}  sm_{cap[0]}{cap[1]}  bf16_supported={bf16_ok}")
            if dtype in ("bf16", "bfloat16") and not bf16_ok:
                log("  NOTE: bf16 requested but card lacks native bf16 — it will "
                    "be emulated/upcast (this is the Titan V case).")
        log(f"requested dtype={dtype}  torch.compile={'on' if compile_on else 'off'}"
            f"  (TORCHDYNAMO_DISABLE={os.environ.get('TORCHDYNAMO_DISABLE','0')})")
    except Exception as e:
        log(f"device info unavailable: {e}")


@contextlib.contextmanager
def gpu_sampler(path, device, interval=0.5):
    """Background nvidia-smi sampler -> appends sm%/power/clock rows to `path`.

    Lets you verify the 'sm% high but power low' question directly per run.
    No-op if path is None or device isn't cuda.
    """
    if not path or not device.startswith("cuda"):
        yield
        return
    import subprocess
    import threading
    idx = 0 if ":" not in device else int(device.split(":")[1])
    stop = threading.Event()

    def _poll():
        with open(path, "a", encoding="utf-8") as f:
            f.write("# ts,gpu_util%,mem_util%,power_W,sm_clock_MHz\n")
            while not stop.is_set():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,utilization.memory,"
                         "power.draw,clocks.sm",
                         "--format=csv,noheader,nounits", "-i", str(idx)],
                        text=True).strip()
                    f.write(f"{time.time():.2f},{out.replace(', ', ',')}\n")
                    f.flush()
                except Exception:
                    pass
                stop.wait(interval)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def wav_duration_seconds(path):
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def save_wav(path, audio, sr):
    """Save a float32/float64 numpy array (mono) to 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.asarray(audio, dtype=np.float32).squeeze()
    # guard against clipping
    peak = np.max(np.abs(a)) if a.size else 0.0
    if peak > 1.0:
        a = a / peak
    pcm = (a * 32767.0).astype("<i2")
    with contextlib.closing(wave.open(path, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def to_numpy(x):
    """Coerce a torch tensor or array-like to a 1-D float numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32).squeeze()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32).squeeze()


# -----------------------------------------------------------------------------
# Speaker similarity (ECAPA-TDNN via SpeechBrain)
# -----------------------------------------------------------------------------
class SpeakerSim:
    """Wraps speechbrain ECAPA-TDNN for cosine speaker similarity.

    Embeds 16 kHz mono audio; compares against a fixed reference embedding.
    """

    def __init__(self, device="cpu", savedir="pretrained_models/ecapa"):
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as e:
            raise SystemExit(
                "speaker similarity needs torch + speechbrain:\n"
                "  pip install speechbrain torchaudio\n"
                f"(import failed: {e})"
            )
        self._torch = __import__("torch")
        self.clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": device},
        )
        self._ref_emb = None

    def _embed_file(self, path):
        import torchaudio
        sig, fs = torchaudio.load(path)
        if sig.shape[0] > 1:               # downmix to mono
            sig = sig.mean(dim=0, keepdim=True)
        if fs != 16000:                    # ECAPA expects 16 kHz
            sig = torchaudio.transforms.Resample(fs, 16000)(sig)
        with self._torch.no_grad():
            emb = self.clf.encode_batch(sig)   # (1, 1, D)
        return emb.squeeze().detach().cpu().numpy()

    def set_reference(self, ref_wav):
        self._ref_emb = self._embed_file(ref_wav)

    def similarity(self, wav_path):
        if self._ref_emb is None:
            raise RuntimeError("call set_reference() first")
        emb = self._embed_file(wav_path)
        a, b = self._ref_emb, emb
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)   # cosine in [-1, 1]


# -----------------------------------------------------------------------------
# VoxCPM2 wrapper
# -----------------------------------------------------------------------------
# If the streaming clone signature differs in your VoxCPM2 build, adjust the
# kwargs assembled in synth_streaming() — that's the one place to touch.
class VoxWrapper:
    def __init__(self, model_id, device="cuda", load_denoiser=False, dtype="auto"):
        try:
            import torch
            from voxcpm import VoxCPM
        except Exception as e:
            raise SystemExit(
                "could not import voxcpm — install it in this env:\n"
                "  pip install voxcpm\n"
                f"(import failed: {e})"
            )
        self._torch = torch
        torch_dtype = {
            "auto": None,
            "fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        }.get(dtype.lower(), None)

        log(f"loading {model_id} on {device} (dtype={dtype}) ...")
        # from_pretrained's exact signature varies across versions and the dtype
        # kwarg name isn't documented, so try the likely ones in order, then
        # fall back to a post-load .to(dtype) cast. Whatever lands, we log it.
        model = None
        attempts = []
        base = dict(load_denoiser=load_denoiser)
        if torch_dtype is not None:
            attempts += [
                {**base, "torch_dtype": torch_dtype},
                {**base, "dtype": torch_dtype},
            ]
        attempts += [base, {}]  # no-dtype, then bare
        last_err = None
        for kw in attempts:
            try:
                model = VoxCPM.from_pretrained(model_id, **kw)
                if "torch_dtype" in kw or "dtype" in kw:
                    log(f"  loaded with explicit dtype kwarg: {list(kw)}")
                break
            except TypeError as e:
                last_err = e
                continue
        if model is None:
            raise SystemExit(f"VoxCPM.from_pretrained failed for all kwarg "
                             f"variants; last error: {last_err}")
        self.model = model

        # If an explicit dtype was requested but the kwarg wasn't accepted, try a
        # best-effort cast of the underlying module. Wrapped in try since not all
        # submodules tolerate casting (the AudioVAE in particular may want fp32).
        if torch_dtype is not None:
            inner = getattr(self.model, "tts_model", None)
            cast_target = None
            for attr in ("model", "net", "backbone"):
                if inner is not None and hasattr(inner, attr):
                    cast_target = getattr(inner, attr)
                    break
            if cast_target is None:
                cast_target = inner
            try:
                if cast_target is not None and hasattr(cast_target, "to"):
                    cast_target.to(torch_dtype)
                    log(f"  cast inner module to {torch_dtype} (best-effort)")
            except Exception as e:
                log(f"  WARNING: dtype cast skipped ({e}); using loaded dtype")

        self.sr = int(getattr(self.model, "tts_model").sample_rate)
        log(f"model sample rate: {self.sr} Hz")

    def _clone_kwargs(self, mode, ref_wav, ref_text):
        """Build the cloning kwargs for a given mode."""
        if mode == "ultimate":
            # reference audio + its transcript (audio-continuation cloning)
            return dict(prompt_wav_path=ref_wav, prompt_text=ref_text,
                        reference_wav_path=ref_wav)
        elif mode == "controllable":
            # reference audio only
            return dict(reference_wav_path=ref_wav)
        else:
            raise ValueError(f"unknown mode: {mode}")

    def synth(self, text, mode, ref_wav, ref_text, cfg_value=2.0,
              inference_timesteps=10):
        """Non-streaming generate. Returns (audio_np, gen_wall_seconds)."""
        kwargs = self._clone_kwargs(mode, ref_wav, ref_text)
        t0 = time.perf_counter()
        wav = self.model.generate(
            text=text, cfg_value=cfg_value,
            inference_timesteps=inference_timesteps, **kwargs,
        )
        dt = time.perf_counter() - t0
        return to_numpy(wav), dt

    def synth_streaming(self, text, mode, ref_wav, ref_text, cfg_value=2.0,
                        inference_timesteps=10):
        """Streaming generate. Returns (audio_np, total_wall_s, first_chunk_s).

        Tries to pass clone kwargs to generate_streaming(); if the build doesn't
        accept them, raises with a clear hint rather than silently dropping the
        reference (which would make 'similarity' meaningless).
        """
        kwargs = self._clone_kwargs(mode, ref_wav, ref_text)
        chunks = []
        t0 = time.perf_counter()
        first_chunk_s = None
        try:
            gen = self.model.generate_streaming(
                text=text, cfg_value=cfg_value,
                inference_timesteps=inference_timesteps, **kwargs,
            )
            for chunk in gen:
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t0
                chunks.append(to_numpy(chunk))
        except TypeError as e:
            raise SystemExit(
                "generate_streaming() rejected the clone kwargs "
                f"({list(kwargs)}). Your VoxCPM2 build likely uses a different "
                "streaming signature. Inspect it with:\n"
                "  python -c \"from voxcpm import VoxCPM; "
                "import inspect; print(inspect.signature("
                "VoxCPM.generate_streaming))\"\n"
                "then edit synth_streaming() / _clone_kwargs() accordingly.\n"
                f"(original error: {e})"
            )
        total = time.perf_counter() - t0
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return audio, total, (first_chunk_s if first_chunk_s is not None else total)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 cloning quality + speed eval")
    ap.add_argument("--ref", default="tests/Alice_VoiceSample.wav",
                    help="reference voice wav (clone source)")
    ap.add_argument("--ref-text", default="tests/Alice_VoiceSample.txt",
                    help="transcript of the reference wav (for 'ultimate' mode)")
    ap.add_argument("--sentences", default=None,
                    help="text file, one sentence per line (else built-in set)")
    ap.add_argument("--model", default="openbmb/VoxCPM2")
    ap.add_argument("--device", default="cuda",
                    help="cuda | cpu | cuda:N (Titan V: try cuda, fall back cpu)")
    ap.add_argument("--modes", default="ultimate,controllable",
                    help="comma list of modes to run")
    ap.add_argument("--streaming", action="store_true",
                    help="also run streaming + measure first-chunk latency")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-sim", action="store_true",
                    help="skip ECAPA similarity (speed-only run)")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16",
                             "float32", "float16", "bfloat16"],
                    help="model precision. Titan V: fp32/fp16 (no native bf16). "
                         "A6000: bf16 runs on hardware. 'auto' lets VoxCPM pick.")
    # --compile / --no-compile are read in _early_compile_setup() before torch
    # imports; declared here so they appear in --help and argparse accepts them.
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile (Ampere+; fuses AR kernel "
                         "launches — usually the biggest speedup). Default off.")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile (default; required on Volta).")
    ap.add_argument("--gpu-log", default=None,
                    help="append a periodic nvidia-smi sm%%/power/clock sample "
                         "to this file during the run (background).")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        raise SystemExit(f"reference wav not found: {args.ref}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    ref_text = ""
    if os.path.isfile(args.ref_text):
        ref_text = read_text(args.ref_text)
    elif "ultimate" in modes:
        raise SystemExit(
            f"'ultimate' mode needs the reference transcript at {args.ref_text}")

    if args.sentences and os.path.isfile(args.sentences):
        sentences = [ln.strip() for ln in open(args.sentences, encoding="utf-8")
                     if ln.strip()]
    else:
        sentences = DEFAULT_SENTENCES
        log(f"using {len(sentences)} built-in sentences (no --sentences given)")

    compile_on = args.compile and not args.no_compile
    log_device_info(args.device, args.dtype, compile_on)

    vox = VoxWrapper(args.model, device=args.device, dtype=args.dtype)

    sim = None
    if not args.no_sim:
        # run ECAPA on CPU by default to keep VRAM clear for VoxCPM on the GPU
        sim_device = "cpu"
        log(f"loading ECAPA-TDNN speaker model on {sim_device} ...")
        sim = SpeakerSim(device=sim_device)
        sim.set_reference(args.ref)
        ref_self = sim.similarity(args.ref)   # sanity: should be ~1.0
        log(f"reference self-similarity (sanity check, ~1.0): {ref_self:.4f}")

    rows = []
    with gpu_sampler(args.gpu_log, args.device):
      for mode in modes:
        for i, text in enumerate(sentences):
            tag = f"{i:02d}"
            # ---- non-streaming ----
            audio, gen_s = vox.synth(
                text, mode, args.ref, ref_text,
                cfg_value=args.cfg, inference_timesteps=args.timesteps)
            wav_path = os.path.join(args.outdir, mode, f"{tag}.wav")
            save_wav(wav_path, audio, vox.sr)
            audio_s = len(audio) / float(vox.sr) if len(audio) else 0.0
            rtf = (gen_s / audio_s) if audio_s > 0 else float("nan")
            s_val = sim.similarity(wav_path) if sim else float("nan")

            row = dict(mode=mode, idx=tag, audio_s=round(audio_s, 3),
                       gen_s=round(gen_s, 3), rtf=round(rtf, 3),
                       sim=round(s_val, 4), first_chunk_s="",
                       stream_rtf="", text=text)

            # ---- streaming (optional) ----
            if args.streaming:
                s_audio, s_total, first_c = vox.synth_streaming(
                    text, mode, args.ref, ref_text,
                    cfg_value=args.cfg, inference_timesteps=args.timesteps)
                s_path = os.path.join(args.outdir, "streaming", f"{mode}_{tag}.wav")
                save_wav(s_path, s_audio, vox.sr)
                s_audio_s = len(s_audio) / float(vox.sr) if len(s_audio) else 0.0
                s_rtf = (s_total / s_audio_s) if s_audio_s > 0 else float("nan")
                row["first_chunk_s"] = round(first_c, 3)
                row["stream_rtf"] = round(s_rtf, 3)

            rows.append(row)
            extra = (f"  first_chunk={row['first_chunk_s']}s"
                     if args.streaming else "")
            log(f"[{mode} {tag}] rtf={row['rtf']}  sim={row['sim']}"
                f"  ({audio_s:.2f}s audio){extra}")

    # ---- write CSV ----
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    fields = ["device", "dtype", "compile", "mode", "idx", "audio_s", "gen_s",
              "rtf", "first_chunk_s", "stream_rtf", "sim", "text"]
    run_cfg = dict(device=args.device, dtype=args.dtype,
                   compile=("on" if compile_on else "off"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=fields)
        wri.writeheader()
        for r in rows:
            wri.writerow({**run_cfg, **{k: r.get(k, "") for k in fields
                                        if k not in run_cfg}})

    # ---- summary ----
    log("=" * 60)
    for mode in modes:
        mrows = [r for r in rows if r["mode"] == mode]
        if not mrows:
            continue
        mean_rtf = np.nanmean([r["rtf"] for r in mrows])
        mean_sim = np.nanmean([r["sim"] for r in mrows]) if sim else float("nan")
        line = f"{mode:13s}  mean RTF={mean_rtf:.3f}"
        if sim:
            line += f"  mean SIM={mean_sim:.4f}"
        if args.streaming:
            mean_fc = np.nanmean([r["first_chunk_s"] for r in mrows
                                  if r["first_chunk_s"] != ""])
            line += f"  mean first_chunk={mean_fc:.3f}s"
        log(line)
    log(f"wrote {csv_path}")
    log("RTF < 1.0 = faster than real time. SIM ~0.7+ = same-speaker territory.")


if __name__ == "__main__":
    main()