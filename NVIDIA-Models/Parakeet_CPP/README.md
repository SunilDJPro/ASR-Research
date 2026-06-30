# Parakeet_CPP — Testing & Evaluation of parakeet.cpp

A workspace for evaluating [parakeet.cpp](https://github.com/mudler/parakeet.cpp)
(the LocalAI C++/ggml inference port of NVIDIA NeMo Parakeet ASR) for both
**offline transcription** and **real-time cache-aware streaming** on this
workstation.

Two evaluation tracks:

1. **Offline / batch** — transcribe the `.wav` files in `tests/` with an offline
   Parakeet model (CTC / TDT / hybrid) and check accuracy + speed.
2. **Real-time streaming** — drive the cache-aware **EOU** model
   (`parakeet_realtime_eou_120m-v1`) live from a microphone, or replay a WAV
   through it chunk-by-chunk to simulate a stream.

> Why this engine: parakeet.cpp runs Parakeet with **no Python at inference**,
> ships a single `libparakeet.so` behind a flat C-API, and produces transcripts
> byte-identical to NeMo (WER 0) while being faster on CPU and GPU. It's a much
> lighter path than a full NeMo + PyTorch stack for benchmarking.

---

## Directory layout

```
Parakeet_CPP/
├── README.md                  <- this file
├── parakeet.cpp/              <- cloned upstream repo (build happens here)
├── models/                    <- downloaded .gguf model weights
├── python/
│   └── parakeet_stream.py     <- ctypes streaming wrapper (file + live mic)
└── tests/
    ├── Alice_VoiceSample.wav
    └── Sunil_TrsDemoEngwav.wav
```

---

## 0. Hardware notes (read first)

This affects which build you make.

- **CPU build is the recommended starting point.** The streaming EOU model is
  only ~120M parameters; CPU comfortably hits the model's target 80–160 ms
  latency in real time. Upstream's own note: GPU gains are *smallest* on the
  encoder-heavy models because ggml's generic CUDA kernels still trail NeMo's
  tuned cuDNN. For a 120M streaming model, CPU is the pragmatic choice.

- **Titan V is Volta = compute capability sm_70.** The prebuilt CUDA release
  bundles target **sm_75 (Turing) and newer**, so they will **not** cover the
  Titan V. If you want GPU on the Titan V you must **build the CUDA backend from
  source** targeting sm_70 (see the optional CUDA section). The A6000 (Ampere,
  sm_86) is covered by prebuilt bundles when it frees up.

- **Audio I/O for live testing:** plugging earphones/mic into whichever box runs
  the test is fine. The streaming path needs **16 kHz, mono, float32** input;
  the wrapper resamples WAV files automatically, and for live capture we use
  `sounddevice` configured to deliver float32 frames.

---

## 1. Prerequisites

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y git build-essential cmake ffmpeg libsndfile1 \
                    portaudio19-dev python3-venv python3-pip
```

- `cmake` ≥ 3.18, a C++17 compiler (gcc/clang).
- `ffmpeg` — handy for converting arbitrary audio to 16 kHz mono WAV.
- `portaudio19-dev` — needed only for live microphone capture via `sounddevice`.
- A Python venv — needed **only** for (a) optional model conversion and (b) the
  live-mic streaming wrapper. **Not** needed for plain CLI inference.

---

## 2. Clone & build parakeet.cpp

From inside `Parakeet_CPP/`:

```bash
git clone --recursive https://github.com/mudler/parakeet.cpp
cd parakeet.cpp
```

> `--recursive` matters: ggml is vendored as a submodule under
> `third_party/ggml`. If you forget it: `git submodule update --init --recursive`.

### 2a. CPU build (recommended first)

Build the CLI, the shared library (for the Python wrapper), and the server:

```bash
cmake -B build \
  -DPARAKEET_BUILD_CLI=ON \
  -DPARAKEET_SHARED=ON \
  -DPARAKEET_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Artifacts after a successful build:

- `build/examples/cli/parakeet-cli`        — the transcribe / stream / info CLI
- `build/examples/server/parakeet-server`  — OpenAI-compatible HTTP server
- `build/libparakeet.so` (or under `build/src/`) — shared lib for FFI

Confirm the C-API symbols the Python wrapper needs are exported:

```bash
nm -D build/libparakeet.so | grep parakeet_capi_stream
# expect: parakeet_capi_stream_begin / _feed / _finalize / _free
```

> If `libparakeet.so` is not at `build/libparakeet.so`, find it with
> `find build -name 'libparakeet.so'` and use that path below.

### 2b. (Optional) Prebuilt binaries — quickest, but no Titan V GPU

If you'd rather skip compiling for **CPU-only** testing, every release ships
prebuilt `parakeet-cli` bundles for Linux/macOS/Windows on the
[Releases page](https://github.com/mudler/parakeet.cpp/releases). Note these do
**not** give you the Titan V GPU path (sm_70) and may not include
`libparakeet.so` for the Python wrapper — build from source if you need either.

### 2c. (Optional) CUDA build for the Titan V (sm_70)

Only if you specifically want to benchmark GPU on the Titan V. Requires the CUDA
toolkit installed:

```bash
cmake -B build-cuda \
  -DPARAKEET_BUILD_CLI=ON \
  -DPARAKEET_SHARED=ON \
  -DPARAKEET_GGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

`-DCMAKE_CUDA_ARCHITECTURES=70` targets Volta. Benchmark this against the CPU
build before assuming it's faster — for the 120M model it may not be.

---

## 3. Download models

Models are GGUF files from the
[`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf)
collection. **F16 is the recommended default** (same accuracy as F32, ~1.7×
smaller, often fastest on CPU).

```bash
pip install -U "huggingface_hub[cli]"

# Streaming EOU model (for real-time testing) -> models/
huggingface-cli download mudler/parakeet-cpp-gguf realtime_eou_120m-v1-f16.gguf \
  --local-dir models/

# An offline model for the .wav accuracy/speed track.
# Hybrid TDT+CTC 0.6B is a good general pick (punctuation + capitalization):
huggingface-cli download mudler/parakeet-cpp-gguf tdt_ctc-110m-f16.gguf \
  --local-dir models/
```

> **Filename caveat:** the exact GGUF filenames in the repo follow a
> `<model>-<quant>.gguf` pattern (e.g. `tdt_ctc-110m-f16.gguf`). Confirm the
> precise names before downloading — list them with:
> ```bash
> huggingface-cli download mudler/parakeet-cpp-gguf --include "*.gguf" \
>   --local-dir /tmp/pk-list --dry-run
> ```
> or just browse the "Files" tab on the HF page. If a name differs, substitute
> it; everything below refers to whatever you saved into `models/`.

**Model choice for your two-track eval:**

| Track | Model | Why |
|---|---|---|
| Offline accuracy on `tests/*.wav` | `parakeet-tdt-0.6b-v2` or `tdt_ctc-110m` | strong WER; hybrid gives punctuation/caps |
| Offline, English, CTC baseline | `parakeet-ctc-0.6b` | the CTC model from your original comparison |
| Real-time streaming | `parakeet_realtime_eou_120m-v1` | the only cache-aware streaming + EOU model |

> The plain **CTC-0.6b is offline-only** in this engine — the `--stream` path and
> the streaming C-API are specific to the EOU model. For real-time, use the EOU
> model.

---

## 4. Offline transcription — the `tests/*.wav` track

Basic transcribe:

```bash
cd parakeet.cpp
./build/examples/cli/parakeet-cli transcribe \
  --model ../models/tdt_ctc-110m-f16.gguf \
  --input ../tests/Alice_VoiceSample.wav
```

With per-word timestamps + confidence:

```bash
./build/examples/cli/parakeet-cli transcribe \
  --model ../models/tdt_ctc-110m-f16.gguf \
  --input ../tests/Sunil_TrsDemoEngwav.wav \
  --timestamps
```

Machine-readable JSON (text + per-word + per-token timestamps/confidence):

```bash
./build/examples/cli/parakeet-cli transcribe \
  --model ../models/tdt_ctc-110m-f16.gguf \
  --input ../tests/Alice_VoiceSample.wav \
  --json > ../alice_result.json
```

Inspect model metadata:

```bash
./build/examples/cli/parakeet-cli info ../models/tdt_ctc-110m-f16.gguf
```

> **Input format:** the CLI loads WAV and resamples to 16 kHz internally. If a
> test file misbehaves, normalize it first:
> ```bash
> ffmpeg -i tests/Sunil_TrsDemoEngwav.wav -ar 16000 -ac 1 -c:a pcm_s16le \
>        tests/Sunil_16k_mono.wav
> ```
> (The second filename looks like it may be missing a dot before `wav` —
> double-check it's a valid `.wav`.)

### Measuring it (accuracy + speed)

- **Speed (RTFx):** time the run and divide audio duration by wall time.
  ```bash
  AUDIO_SEC=$(ffprobe -v error -show_entries format=duration \
              -of default=nk=1:nw=1 ../tests/Alice_VoiceSample.wav)
  /usr/bin/time -v ./build/examples/cli/parakeet-cli transcribe \
    --model ../models/tdt_ctc-110m-f16.gguf \
    --input ../tests/Alice_VoiceSample.wav 2> ../alice_timing.txt
  # RTFx = AUDIO_SEC / elapsed_wall_seconds  (higher = faster than real time)
  ```
- **Accuracy (WER):** you need a ground-truth transcript per file. Put
  references in `tests/Alice_VoiceSample.ref.txt` etc., then compute WER with
  `jiwer`:
  ```bash
  pip install jiwer
  python - <<'PY'
  import jiwer, json, pathlib
  hyp = json.load(open("../alice_result.json"))["text"]
  ref = pathlib.Path("../tests/Alice_VoiceSample.ref.txt").read_text().strip()
  print("WER:", jiwer.wer(ref, hyp))
  PY
  ```
  Normalize case/punctuation consistently between ref and hyp before scoring
  (lowercase, strip punctuation) — especially when comparing a CTC model
  (lowercase, no punctuation) against a hybrid model (mixed case + punctuation).

---

## 5. Real-time streaming — the EOU track

### 5a. Replay a WAV as a simulated stream (CLI)

```bash
cd parakeet.cpp
./build/examples/cli/parakeet-cli transcribe \
  --model ../models/realtime_eou_120m-v1-f16.gguf \
  --input ../tests/Alice_VoiceSample.wav \
  --stream --timestamps
```

This feeds the clip on the model's chunk schedule and prints partial text
incrementally, plus `[EOU @ <t>s]` / `[EOB @ <t>s]` markers, then the finalized
tail. Good for sanity-checking streaming output against the offline transcript.

### 5b. Live microphone (Python wrapper)

Set up the venv and deps:

```bash
cd ../python    # Parakeet_CPP/python
python3 -m venv .venv
source .venv/bin/activate
pip install sounddevice numpy
```

Point the wrapper at your built shared library:

```bash
export PARAKEET_LIB="$(find ../parakeet.cpp/build -name libparakeet.so | head -1)"
```

Replay a WAV through the wrapper (no mic needed):

```bash
python parakeet_stream.py \
  ../models/realtime_eou_120m-v1-f16.gguf \
  ../tests/Alice_VoiceSample.wav
```

For **live mic** capture, see the `sounddevice` callback example at the bottom
of `parakeet_stream.py` — it opens the default input device at 16 kHz mono
float32 and feeds frames into a `StreamSession` as they arrive, printing
finalized text and EOU events. List/select your audio device with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
# then set the device index/env as noted in the script
```

> **Audio device tip:** whether you plug earphones into the AI server or use the
> Titan V workstation's existing audio I/O, just make sure the chosen input
> device shows up in `query_devices()` and is set as default (or passed
> explicitly). Capture at 16 kHz mono if the device allows; otherwise let
> `sounddevice`/the wrapper resample.

---

## 6. (Optional) OpenAI-compatible server

`parakeet-server` exposes an OpenAI transcription API, so any OpenAI client
works by pointing `base_url` at it — useful if you want to benchmark from a
Python harness without FFI:

```bash
./build/examples/server/parakeet-server \
  --model ../models/tdt_ctc-110m-f16.gguf
# then POST audio to the /v1/audio/transcriptions endpoint it prints
```

---

## 7. Suggested evaluation checklist

- [ ] CPU build succeeds; `parakeet-cli info` reads a model.
- [ ] Offline transcribe of both `tests/*.wav` produces sensible text.
- [ ] Record RTFx for each offline model on the Titan V WS (CPU).
- [ ] Compute WER vs. your reference transcripts (normalize first).
- [ ] EOU streaming replay matches the offline transcript closely.
- [ ] Live-mic streaming produces stable partials + `[EOU]` at utterance ends.
- [ ] (Optional) CUDA sm_70 build — compare GPU vs CPU RTFx on the Titan V.
- [ ] (Optional) Repeat offline accuracy on the A6000 when free.

---

## Troubleshooting

- **`libparakeet.so` not found by the wrapper** — set `PARAKEET_LIB` to the
  absolute path (see §5b), or `ldconfig`/`LD_LIBRARY_PATH` the build dir.
- **Submodule/ggml build errors** — `git submodule update --init --recursive`.
- **Empty streaming output** — the EOU model needs ≥160 ms of audio before it
  emits anything, and only speech produces tokens; silence yields `""`.
- **Garbled transcript** — verify input really is mono 16 kHz; re-encode with
  the `ffmpeg` line in §4.
- **CUDA build can't find Volta** — confirm `-DCMAKE_CUDA_ARCHITECTURES=70` and a
  CUDA toolkit new enough to still emit sm_70.
- **CLI flag differences** — upstream evolves; if a flag here is rejected, check
  `parakeet-cli transcribe --help` and the repo README for the current syntax.

---

## References

- parakeet.cpp: https://github.com/mudler/parakeet.cpp
- GGUF weights: https://huggingface.co/mudler/parakeet-cpp-gguf
- EOU streaming model card: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1