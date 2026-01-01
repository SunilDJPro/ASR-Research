# NVIDIA Canary-Qwen-2.5B ASR Evaluation Playground

A production-grade evaluation script for testing NVIDIA's Canary-Qwen-2.5B speech recognition model.

## Metrics Measured

| Metric | Description |
|--------|-------------|
| **RTF** | Real-Time Factor (inference_time / audio_duration). Lower = faster |
| **RTFx** | Inverse of RTF. How many times faster than real-time |
| **First Token Latency** | Time until first transcription output (important for streaming) |
| **VRAM Usage** | Peak and average GPU memory consumption |
| **GPU Utilization** | Peak and average GPU compute utilization |

## Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (Titan V, V100, A6000, etc.)
- **CUDA**: CUDA 12.x recommended
- **Python**: 3.10+
- **PyTorch**: 2.6+ (for FSDP2 support required by NeMo)

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv canary_eval
source canary_eval/bin/activate

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install NeMo toolkit (latest trunk required for SALM)
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"

# 4. Install additional dependencies
pip install librosa soundfile py3nvml

# Or if not using venv:
pip install --break-system-packages librosa soundfile py3nvml
pip install --break-system-packages "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```

## Usage

```bash
python canary_asr_eval.py
```

The script will:
1. Load the model (~5GB download on first run)
2. Run a warmup inference
3. Enter interactive mode - paste audio file paths to transcribe

### Example Session

```
╔══════════════════════════════════════════════════════════════╗
║     NVIDIA Canary-Qwen-2.5B ASR Evaluation Playground        ║
╚══════════════════════════════════════════════════════════════╝

GPU Detected: NVIDIA TITAN V
Total VRAM: 12288 MB
Loading NVIDIA Canary-Qwen-2.5B...
Model loaded in 45.23s
Running warmup inference...
Warmup complete. Peak VRAM: 8234 MB

Ready for inference!
Enter audio file path (.wav) or 'q' to quit

🎤 Audio file path: /path/to/recording.wav

────────────────────────────────────────────────────────────────
FILE: /path/to/recording.wav
────────────────────────────────────────────────────────────────

📊 AUDIO INFO:
   Duration:        15.32s
   Original SR:     24000 Hz
   Chunks:          1

⏱️  TIMING:
   Preprocessing:   45.2 ms
   Inference:       892.3 ms
   Total:           937.5 ms
   First Token:     892.3 ms

🚀 PERFORMANCE:
   RTF:             0.0582 (lower is better)
   RTFx:            17.2x faster than real-time
   Throughput:      17.17 sec audio/sec

🖥️  GPU UTILIZATION:
   VRAM Peak:       8456 / 12288 MB (68.8%)
   VRAM Avg:        8234 MB
   GPU Util Peak:   98%
   GPU Util Avg:    72.3%

📝 TRANSCRIPT:
   Hello, this is a test recording for evaluating
   the speech recognition capabilities...
────────────────────────────────────────────────────────────────
```

## Audio Format Notes

- **Input**: .wav, .flac, .mp3, .ogg, .m4a (auto-resampled to 16kHz mono)
- **Your files**: 24kHz mono → automatically resampled to 16kHz
- **Max chunk**: 30 seconds (longer files are chunked automatically)

## Understanding the Metrics

### RTF (Real-Time Factor)
- `RTF = inference_time / audio_duration`
- RTF < 1.0 means faster than real-time
- Example: RTF = 0.05 means 1 second of audio takes 50ms to process

### RTFx (Real-Time Factor X)
- `RTFx = 1 / RTF` 
- How many times faster than real-time
- Example: RTFx = 20 means processing is 20x faster than audio duration
- NVIDIA claims 418 RTFx on optimized hardware

### First Token Latency
- Time from inference start to first output token
- Critical for streaming/real-time applications
- Lower = more responsive user experience

## Troubleshooting

### CUDA Out of Memory
- The 2.5B model needs ~8-10GB VRAM during inference
- 12GB Titan V should work, but long audio may cause issues
- Try shorter audio clips or use A6000 (48GB)

### Model Download Issues
- First run downloads ~5GB from HuggingFace
- Ensure stable internet connection
- Model cached at `~/.cache/huggingface/`

### NeMo Import Errors
- Ensure PyTorch 2.6+ is installed
- Use the git install method for latest NeMo trunk

## Session Summary

On exit (Ctrl+C), the script prints aggregated statistics for all processed files:
- Total audio processed
- Average RTFx across all files  
- Average first token latency
