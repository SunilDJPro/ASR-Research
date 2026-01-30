#!/bin/bash
# NVIDIA PersonaPlex Demo - Setup Script
# Run with: bash setup.sh

set -e

echo "=========================================="
echo "NVIDIA PersonaPlex Demo Setup"
echo "=========================================="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "[WARNING] HF_TOKEN environment variable is not set!"
    echo "Please set it before running the demo:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo ""
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "[✓] NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[!] nvidia-smi not found - GPU may not be available"
fi

# Install system dependencies
echo ""
echo "[1/4] Installing system dependencies..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y libopus-dev ffmpeg
    echo "[✓] System dependencies installed"
else
    echo "[!] apt not found - please install libopus-dev and ffmpeg manually"
fi

# Create virtual environment (optional)
echo ""
echo "[2/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[✓] Virtual environment created"
fi

# Activate venv and install dependencies
echo ""
echo "[3/4] Installing Python dependencies..."
source venv/bin/activate

pip install --upgrade pip

# Install PersonaPlex from NVIDIA repo
echo "Installing PersonaPlex from GitHub..."
pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi

# Install other requirements
pip install -r requirements.txt

echo "[✓] Python dependencies installed"

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchaudio; print(f'Torchaudio version: {torchaudio.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure HF_TOKEN is set: export HF_TOKEN=your_token"
echo "  2. Accept model license: https://huggingface.co/nvidia/personaplex-7b-v1"
echo "  3. Add audio files to assets/ folder:"
echo "     - main_input.wav (user speech input)"
echo "     - interruption.wav (for barge-in test)"
echo "     - sample_voice.wav (for voice cloning)"
echo "  4. Activate venv: source venv/bin/activate"
echo "  5. Run demo: python main_script.py --mode all"
echo ""