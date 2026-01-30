# NVIDIA PersonaPlex-7B-v1 Demo

Full-duplex speech-to-speech conversational AI with custom voice cloning, configurable personas, and interruption handling.

## Features

| Feature | Description |
|---------|-------------|
| **Voice Cloning** | Create custom voice embeddings from any English audio sample |
| **Conversation** | Run end-to-end speech-to-speech inference with configurable text prompts |
| **Interruption Handling** | Test full-duplex barge-in capability with overlaid interruption audio |
| **Metrics Collection** | Latency, RTF (real-time factor), GPU memory usage |

## Requirements

### Hardware
- NVIDIA GPU with 48GB+ VRAM (tested on A6000)
- CUDA 12.x compatible

### Software
- Ubuntu 24.04 (or similar Linux)
- Python 3.10+
- libopus-dev system library

## Quick Start

### 1. System Dependencies

```bash
sudo apt update
sudo apt install libopus-dev ffmpeg
```

### 2. Accept Model License

Visit [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) and accept the license.

### 3. Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 4. Install Python Dependencies

```bash
# Install PersonaPlex from NVIDIA repo
pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi

# Install other dependencies
pip install -r requirements.txt
```

### 5. Prepare Audio Files

Place your audio files in `assets/`:

```
assets/
├── main_input.wav       # Primary input for conversation (user speech)
├── interruption.wav     # Interruption audio for barge-in test
└── sample_voice.wav     # Reference audio for voice cloning
```

**Audio Requirements:**
- Format: WAV (will auto-convert to 24kHz mono if needed)
- Content: Clear English speech
- Duration: 5-30 seconds recommended for voice sample

### 6. Run the Demo

```bash
# Run all demos (voice encoding + conversation + interruption)
python main_script.py --mode all

# Run specific demos
python main_script.py --mode voice           # Only create voice embedding
python main_script.py --mode conversation    # Only run conversation
python main_script.py --mode interruption    # Only test interruption handling
```

## Usage Examples

### Basic Usage

```bash
python main_script.py --mode all
```

### Custom Text Prompt

```bash
python main_script.py --mode conversation \
    --text-prompt "You are a technical interviewer asking about machine learning experience."
```

### Custom Interruption Timing

```bash
python main_script.py --mode interruption --interrupt-delay 3.5
```

### Use Pre-saved Voice Embedding

```bash
python main_script.py --mode conversation --voice-embed output/custom_voice.pt
```

### Use Predefined NVIDIA Voices

If you want to use NVIDIA's pre-packaged voices instead of custom cloning:

```bash
# Download from HF repo (after model loads, voices are in cache)
# Available: NATF0-3 (female), NATM0-3 (male), VARF0-4, VARM0-4
python main_script.py --mode conversation --voice-embed NATF2.pt
```

## Output Files

After running, check `output/`:

```
output/
├── custom_voice.pt                     # Your encoded voice embedding
├── response.wav                        # Generated response audio
├── transcription.json                  # Text tokens and metadata
├── response_with_interruption.wav      # Response from interruption test
├── transcription_interruption.json     # Transcription from interruption test
├── mixed_input_with_interruption.wav   # Debug: combined input audio
└── metrics.json                        # Performance metrics
```

## Text Prompt Guide

PersonaPlex supports various personas through text prompts (max ~200 tokens):

### Interview Style (Default)
```
You enjoy having a good conversation. You are conducting a casual, friendly interview. 
Ask thoughtful follow-up questions and share your own perspectives when appropriate.
```

### Assistant Role
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear 
and engaging way.
```

### Customer Service
```
You work for TechSupport Pro and your name is Alex. Information: Verify customer 
account number. Available hours: 9am-5pm EST. Premium support: $29/month add-on.
```

### Casual Conversation
```
You enjoy having a good conversation. Have a reflective conversation about travel 
experiences and favorite destinations.
```

## Performance Notes

| Metric | Expected Value (A6000 48GB) |
|--------|----------------------------|
| Model Load Time | ~60-90 seconds |
| Voice Encoding | <1 second |
| Inference RTF | 0.2-0.4x (faster than real-time) |
| GPU Memory | ~30-40 GB |

## Troubleshooting

### "CUDA out of memory"
Add `--cpu-offload` flag or reduce batch size. The A6000 48GB should handle it fine.

### "HF_TOKEN not set"
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Audio sounds choppy
Ensure input audio is clean. Pre-process with:
```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -c:a pcm_s16le -af "lowpass=f=8000" output.wav
```

### Model license error
Visit https://huggingface.co/nvidia/personaplex-7b-v1 and accept the license while logged in.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PersonaPlex Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  Voice Sample → Mimi Encoder → Voice Codes (.pt)            │
│                                                              │
│  Input Audio → Mimi Encoder → Audio Tokens                  │
│                      ↓                                       │
│  [Voice Codes + Audio Tokens + Text Prompt]                 │
│                      ↓                                       │
│            Moshi LM (7B Transformer)                        │
│                      ↓                                       │
│         Text Tokens + Audio Tokens                          │
│                      ↓                                       │
│              Mimi Decoder                                   │
│                      ↓                                       │
│            Output Audio (.wav)                              │
└─────────────────────────────────────────────────────────────┘
```

## References

- [PersonaPlex HuggingFace](https://huggingface.co/nvidia/personaplex-7b-v1)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [NVIDIA Research Page](https://research.nvidia.com/labs/adlr/personaplex/)
- [Moshi Architecture](https://github.com/kyutai-labs/moshi)

## License

This demo script is provided as-is. PersonaPlex model weights are under [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).