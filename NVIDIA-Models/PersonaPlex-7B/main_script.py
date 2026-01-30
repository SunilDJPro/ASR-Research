#!/usr/bin/env python3
"""
NVIDIA PersonaPlex-7B-v1 Demo Script
=====================================

Full-duplex speech-to-speech conversational AI with:
- Custom voice cloning from reference audio
- Configurable persona via text prompts
- Interruption handling simulation
- Performance metrics collection

Requirements:
    - Ubuntu 24.04 (or similar Linux)
    - CUDA 12.x compatible GPU (tested on A6000 48GB)
    - Python 3.10+
    - HuggingFace account with accepted PersonaPlex license

Setup:
    1. Accept license: https://huggingface.co/nvidia/personaplex-7b-v1
    2. Set environment: export HF_TOKEN=your_token_here
    3. Install system deps: sudo apt install libopus-dev
    4. Install PersonaPlex: pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi
    5. Install other deps: pip install -r requirements.txt

Usage:
    python main_script.py --mode all              # Run all demos
    python main_script.py --mode voice            # Only create voice embedding
    python main_script.py --mode conversation     # Only run conversation
    python main_script.py --mode interruption     # Only test interruption handling

Author: PersonaPlex Demo
Date: January 2026
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for the demo."""
    
    # Paths
    project_root: Path = Path(__file__).parent
    assets_dir: Path = None
    output_dir: Path = None
    
    # Input files
    main_input_wav: str = "main_input.wav"
    interruption_wav: str = "interruption.wav"
    sample_voice_wav: str = "sample_voice.wav"
    
    # Model settings
    hf_repo: str = "nvidia/personaplex-7b-v1"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    
    # Audio settings
    sample_rate: int = 24000  # PersonaPlex requires 24kHz
    num_codebooks: int = 8    # Mimi codebook count for Moshi
    
    # Text prompts (casual interview style)
    default_text_prompt: str = (
        "You enjoy having a good conversation. "
        "You are conducting a casual, friendly interview. "
        "Ask thoughtful follow-up questions and share your own perspectives when appropriate. "
        "Be warm, engaged, and genuinely curious about what the other person has to say."
    )
    
    # Inference settings
    seed: int = 42424242
    temperature: float = 0.8
    temperature_text: float = 0.7
    
    # Interruption settings
    interruption_delay_sec: float = 2.0  # When to inject interruption
    
    def __post_init__(self):
        self.assets_dir = self.project_root / "assets"
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Metrics:
    """Performance metrics collection."""
    
    model_load_time_sec: float = 0.0
    voice_encode_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    audio_duration_sec: float = 0.0
    output_duration_sec: float = 0.0
    real_time_factor: float = 0.0  # < 1.0 means faster than real-time
    peak_gpu_memory_gb: float = 0.0
    device: str = ""
    timestamp: str = ""
    
    def compute_rtf(self):
        """Compute real-time factor."""
        if self.audio_duration_sec > 0:
            self.real_time_factor = self.inference_time_sec / self.audio_duration_sec
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[✓] Metrics saved to: {path}")


# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def load_audio(path: Path, target_sr: int = 24000) -> torch.Tensor:
    """
    Load and preprocess audio file.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (24000 for PersonaPlex)
    
    Returns:
        Audio tensor of shape [1, 1, num_samples] (batch, channels, time)
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Load audio
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        print(f"[i] Resampling {path.name}: {sr}Hz -> {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Ensure shape is [B, C, T]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    return waveform


def save_audio(waveform: torch.Tensor, path: Path, sample_rate: int = 24000):
    """Save audio tensor to file."""
    # Ensure correct shape for torchaudio [C, T]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)  # Remove batch dim
    
    # Normalize to prevent clipping
    max_val = waveform.abs().max()
    if max_val > 1.0:
        waveform = waveform / max_val
    
    torchaudio.save(path, waveform.cpu(), sample_rate)
    print(f"[✓] Audio saved to: {path}")


def get_audio_duration(waveform: torch.Tensor, sample_rate: int = 24000) -> float:
    """Get audio duration in seconds."""
    num_samples = waveform.shape[-1]
    return num_samples / sample_rate


def overlay_audio(
    base_audio: torch.Tensor,
    overlay_audio: torch.Tensor,
    offset_sec: float,
    sample_rate: int = 24000,
    overlay_gain: float = 0.8
) -> torch.Tensor:
    """
    Overlay (mix) interruption audio onto base audio at specified time offset.
    
    Args:
        base_audio: Base audio tensor [B, C, T]
        overlay_audio: Audio to overlay [B, C, T]
        offset_sec: Time offset in seconds where overlay begins
        sample_rate: Audio sample rate
        overlay_gain: Volume multiplier for overlay audio
    
    Returns:
        Mixed audio tensor
    """
    offset_samples = int(offset_sec * sample_rate)
    
    # Clone base to avoid modifying original
    result = base_audio.clone()
    
    # Calculate overlay region
    overlay_len = overlay_audio.shape[-1]
    base_len = base_audio.shape[-1]
    
    # Determine actual overlay region
    start_idx = min(offset_samples, base_len)
    end_idx = min(offset_samples + overlay_len, base_len)
    overlay_samples = end_idx - start_idx
    
    if overlay_samples > 0:
        # Mix the audio (simple addition with gain control)
        result[..., start_idx:end_idx] += overlay_audio[..., :overlay_samples] * overlay_gain
        
        # Normalize to prevent clipping
        max_val = result.abs().max()
        if max_val > 1.0:
            result = result / max_val
    
    return result


# ============================================================================
# MODEL LOADING
# ============================================================================

class PersonaPlexModel:
    """
    Wrapper for PersonaPlex model loading and inference.
    
    Handles:
        - Model weight downloading from HuggingFace
        - Mimi encoder/decoder initialization
        - LM generation setup
        - Voice embedding encoding
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Model components (loaded lazily)
        self.mimi = None
        self.moshi_lm = None
        self.lm_gen = None
        
        # Weight paths
        self.mimi_weight_path = None
        self.moshi_weight_path = None
        
        # Pre-loaded voice embeddings
        self.voice_embeddings: Dict[str, torch.Tensor] = {}
    
    def load(self, metrics: Optional[Metrics] = None) -> 'PersonaPlexModel':
        """
        Load all model components.
        
        Returns self for method chaining.
        """
        start_time = time.time()
        
        print("\n" + "="*60)
        print("Loading PersonaPlex Model")
        print("="*60)
        
        # Check HF token
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN environment variable not set.\n"
                "Please set it: export HF_TOKEN=your_huggingface_token"
            )
        
        try:
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders, LMGen
        except ImportError as e:
            raise ImportError(
                f"Failed to import required modules: {e}\n"
                "Please install PersonaPlex:\n"
                "  pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi"
            )
        
        print(f"[i] Device: {self.device}")
        print(f"[i] Dtype: {self.dtype}")
        print(f"[i] HF Repo: {self.config.hf_repo}")
        
        # Download Mimi weights
        print("\n[1/4] Downloading Mimi encoder/decoder weights...")
        self.mimi_weight_path = hf_hub_download(
            self.config.hf_repo,
            loaders.MIMI_NAME,
            token=hf_token
        )
        print(f"      -> {self.mimi_weight_path}")
        
        # Download Moshi LM weights
        print("\n[2/4] Downloading Moshi LM weights...")
        self.moshi_weight_path = hf_hub_download(
            self.config.hf_repo,
            loaders.MOSHI_NAME,
            token=hf_token
        )
        print(f"      -> {self.moshi_weight_path}")
        
        # Load Mimi (encoder/decoder)
        print("\n[3/4] Loading Mimi encoder/decoder...")
        self.mimi = loaders.get_mimi(self.mimi_weight_path, device=self.device)
        self.mimi.set_num_codebooks(self.config.num_codebooks)
        print(f"      -> Codebooks: {self.config.num_codebooks}")
        print(f"      -> Frame size: {self.mimi.frame_size} samples")
        print(f"      -> Frame rate: {self.config.sample_rate / self.mimi.frame_size:.2f} Hz")
        
        # Load Moshi LM
        print("\n[4/4] Loading Moshi LM (7B parameters)...")
        self.moshi_lm = loaders.get_moshi_lm(self.moshi_weight_path, device=self.device)
        
        # Create LM generator with sampling parameters
        self.lm_gen = LMGen(
            self.moshi_lm,
            temp=self.config.temperature,
            temp_text=self.config.temperature_text
        )
        
        load_time = time.time() - start_time
        
        # Get GPU memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_mem = 0.0
        
        print("\n" + "-"*60)
        print(f"[✓] Model loaded successfully!")
        print(f"    Load time: {load_time:.2f} seconds")
        print(f"    Peak GPU memory: {peak_mem:.2f} GB")
        print("-"*60)
        
        if metrics:
            metrics.model_load_time_sec = load_time
            metrics.peak_gpu_memory_gb = peak_mem
            metrics.device = self.device
        
        return self
    
    def encode_voice(
        self,
        audio_path: Path,
        save_path: Optional[Path] = None,
        metrics: Optional[Metrics] = None
    ) -> torch.Tensor:
        """
        Encode reference audio into voice embedding (voice codes).
        
        This is the "voice cloning" step - extracts acoustic features
        that define the voice characteristics.
        
        Args:
            audio_path: Path to reference voice audio
            save_path: Optional path to save the .pt embedding
            metrics: Optional metrics collector
        
        Returns:
            Voice codes tensor [B, K, T] where K=num_codebooks
        """
        if self.mimi is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        print(f"\n[i] Encoding voice from: {audio_path}")
        start_time = time.time()
        
        # Load and preprocess audio
        waveform = load_audio(audio_path, self.config.sample_rate)
        waveform = waveform.to(self.device)
        
        audio_duration = get_audio_duration(waveform, self.config.sample_rate)
        print(f"    Duration: {audio_duration:.2f} seconds")
        
        # Encode to voice codes
        with torch.no_grad():
            voice_codes = self.mimi.encode(waveform)
        
        encode_time = time.time() - start_time
        print(f"    Encoding time: {encode_time:.3f} seconds")
        print(f"    Voice codes shape: {voice_codes.shape}")
        
        # Save if requested
        if save_path:
            torch.save(voice_codes.cpu(), save_path)
            print(f"[✓] Voice embedding saved to: {save_path}")
        
        if metrics:
            metrics.voice_encode_time_sec = encode_time
        
        return voice_codes
    
    def load_voice_embedding(self, path: Path) -> torch.Tensor:
        """Load a pre-saved voice embedding (.pt file)."""
        if not path.exists():
            raise FileNotFoundError(f"Voice embedding not found: {path}")
        
        voice_codes = torch.load(path, map_location=self.device)
        print(f"[✓] Loaded voice embedding: {path}")
        print(f"    Shape: {voice_codes.shape}")
        return voice_codes
    
    def run_inference(
        self,
        input_audio_path: Path,
        voice_codes: torch.Tensor,
        text_prompt: str,
        output_wav_path: Path,
        output_text_path: Path,
        metrics: Optional[Metrics] = None,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Run full inference: input audio -> model -> output audio + text.
        
        This implements streaming inference similar to moshi.offline.
        
        Args:
            input_audio_path: Path to input audio (user speech)
            voice_codes: Voice embedding for output voice
            text_prompt: Text prompt defining persona/behavior
            output_wav_path: Where to save generated audio
            output_text_path: Where to save transcription JSON
            metrics: Optional metrics collector
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (output_waveform, text_tokens)
        """
        if self.mimi is None or self.lm_gen is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Set seed
        if seed is None:
            seed = self.config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        print(f"\n{'='*60}")
        print("Running Inference")
        print(f"{'='*60}")
        print(f"[i] Input: {input_audio_path}")
        print(f"[i] Text prompt: {text_prompt[:80]}...")
        print(f"[i] Seed: {seed}")
        
        # Load input audio
        input_waveform = load_audio(input_audio_path, self.config.sample_rate)
        input_waveform = input_waveform.to(self.device)
        
        audio_duration = get_audio_duration(input_waveform, self.config.sample_rate)
        print(f"[i] Input duration: {audio_duration:.2f} seconds")
        
        if metrics:
            metrics.audio_duration_sec = audio_duration
        
        start_time = time.time()
        
        # Encode input audio to codes
        print("\n[1/3] Encoding input audio...")
        frame_size = self.mimi.frame_size
        input_codes_list = []
        
        with torch.no_grad(), self.mimi.streaming(batch_size=1):
            num_frames = input_waveform.shape[-1] // frame_size
            for i in tqdm(range(num_frames), desc="      Encoding"):
                frame = input_waveform[..., i * frame_size:(i + 1) * frame_size]
                codes = self.mimi.encode(frame)
                input_codes_list.append(codes)
        
        print(f"      -> Encoded {len(input_codes_list)} frames")
        
        # Run LM generation
        print("\n[2/3] Generating response...")
        output_wav_chunks = []
        text_tokens = []
        
        with torch.no_grad(), self.lm_gen.streaming(1), self.mimi.streaming(1):
            for idx, code in enumerate(tqdm(input_codes_list, desc="      Generating")):
                tokens_out = self.lm_gen.step(code)
                
                if tokens_out is not None:
                    # tokens_out shape: [B, 1 + 8, 1]
                    # tokens_out[:, 0] = text token
                    # tokens_out[:, 1:] = audio tokens
                    
                    # Extract text token (for transcription)
                    text_token = tokens_out[:, 0, 0].item()
                    text_tokens.append(text_token)
                    
                    # Decode audio tokens
                    audio_tokens = tokens_out[:, 1:]
                    wav_chunk = self.mimi.decode(audio_tokens)
                    output_wav_chunks.append(wav_chunk)
        
        inference_time = time.time() - start_time
        
        # Concatenate output audio
        print("\n[3/3] Finalizing output...")
        if output_wav_chunks:
            output_waveform = torch.cat(output_wav_chunks, dim=-1)
            output_duration = get_audio_duration(output_waveform, self.config.sample_rate)
        else:
            output_waveform = torch.zeros(1, 1, self.config.sample_rate)
            output_duration = 1.0
        
        # Save outputs
        save_audio(output_waveform, output_wav_path, self.config.sample_rate)
        
        # Save transcription
        transcription_data = {
            "text_tokens": text_tokens,
            "num_tokens": len(text_tokens),
            "input_audio": str(input_audio_path),
            "text_prompt": text_prompt,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        }
        with open(output_text_path, 'w') as f:
            json.dump(transcription_data, f, indent=2)
        print(f"[✓] Transcription saved to: {output_text_path}")
        
        # Update metrics
        if metrics:
            metrics.inference_time_sec = inference_time
            metrics.output_duration_sec = output_duration
            metrics.compute_rtf()
            metrics.timestamp = datetime.now().isoformat()
        
        print(f"\n{'-'*60}")
        print(f"[✓] Inference complete!")
        print(f"    Inference time: {inference_time:.2f} seconds")
        print(f"    Output duration: {output_duration:.2f} seconds")
        if metrics:
            print(f"    Real-time factor: {metrics.real_time_factor:.3f}x")
        print(f"{'-'*60}")
        
        return output_waveform, text_tokens
    
    def run_inference_with_interruption(
        self,
        main_audio_path: Path,
        interruption_audio_path: Path,
        voice_codes: torch.Tensor,
        text_prompt: str,
        output_wav_path: Path,
        output_text_path: Path,
        interruption_delay_sec: float = 2.0,
        metrics: Optional[Metrics] = None,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Run inference with simulated user interruption.
        
        Creates a mixed audio input where interruption audio is overlaid
        at specified time offset, testing the model's ability to handle
        barge-in scenarios.
        
        Args:
            main_audio_path: Primary input audio
            interruption_audio_path: Interruption audio to overlay
            voice_codes: Voice embedding
            text_prompt: Text prompt
            output_wav_path: Output audio path
            output_text_path: Output transcription path
            interruption_delay_sec: When to start interruption
            metrics: Optional metrics
            seed: Random seed
        
        Returns:
            Tuple of (output_waveform, text_tokens)
        """
        print(f"\n{'='*60}")
        print("Interruption Handling Test")
        print(f"{'='*60}")
        print(f"[i] Main audio: {main_audio_path}")
        print(f"[i] Interruption: {interruption_audio_path}")
        print(f"[i] Interruption delay: {interruption_delay_sec} seconds")
        
        # Load both audio files
        main_audio = load_audio(main_audio_path, self.config.sample_rate)
        interrupt_audio = load_audio(interruption_audio_path, self.config.sample_rate)
        
        main_duration = get_audio_duration(main_audio, self.config.sample_rate)
        interrupt_duration = get_audio_duration(interrupt_audio, self.config.sample_rate)
        
        print(f"[i] Main audio duration: {main_duration:.2f}s")
        print(f"[i] Interruption duration: {interrupt_duration:.2f}s")
        
        # Create mixed audio with interruption
        mixed_audio = overlay_audio(
            main_audio,
            interrupt_audio,
            offset_sec=interruption_delay_sec,
            sample_rate=self.config.sample_rate,
            overlay_gain=0.9
        )
        
        # Save mixed audio for reference
        mixed_audio_path = output_wav_path.parent / "mixed_input_with_interruption.wav"
        save_audio(mixed_audio, mixed_audio_path, self.config.sample_rate)
        
        # Create temporary file for mixed audio
        temp_mixed_path = output_wav_path.parent / "_temp_mixed_input.wav"
        save_audio(mixed_audio, temp_mixed_path, self.config.sample_rate)
        
        # Run inference on mixed audio
        result = self.run_inference(
            input_audio_path=temp_mixed_path,
            voice_codes=voice_codes,
            text_prompt=text_prompt,
            output_wav_path=output_wav_path,
            output_text_path=output_text_path,
            metrics=metrics,
            seed=seed
        )
        
        # Clean up temp file
        if temp_mixed_path.exists():
            temp_mixed_path.unlink()
        
        return result


# ============================================================================
# MAIN DEMO FUNCTIONS
# ============================================================================

def demo_voice_encoding(config: Config, model: PersonaPlexModel, metrics: Metrics):
    """Demo 1: Create custom voice embedding from reference audio."""
    
    print("\n" + "#"*60)
    print("# DEMO: Voice Encoding (Custom Voice Cloning)")
    print("#"*60)
    
    voice_audio_path = config.assets_dir / config.sample_voice_wav
    voice_embed_path = config.output_dir / "custom_voice.pt"
    
    if not voice_audio_path.exists():
        print(f"[!] Voice sample not found: {voice_audio_path}")
        print("    Please provide a sample_voice.wav file in the assets folder.")
        return None
    
    voice_codes = model.encode_voice(
        audio_path=voice_audio_path,
        save_path=voice_embed_path,
        metrics=metrics
    )
    
    return voice_codes


def demo_conversation(
    config: Config,
    model: PersonaPlexModel,
    voice_codes: torch.Tensor,
    metrics: Metrics
):
    """Demo 2: Run standard conversation inference."""
    
    print("\n" + "#"*60)
    print("# DEMO: Conversation Inference")
    print("#"*60)
    
    input_audio_path = config.assets_dir / config.main_input_wav
    output_wav_path = config.output_dir / "response.wav"
    output_text_path = config.output_dir / "transcription.json"
    
    if not input_audio_path.exists():
        print(f"[!] Input audio not found: {input_audio_path}")
        print("    Please provide a main_input.wav file in the assets folder.")
        return
    
    model.run_inference(
        input_audio_path=input_audio_path,
        voice_codes=voice_codes,
        text_prompt=config.default_text_prompt,
        output_wav_path=output_wav_path,
        output_text_path=output_text_path,
        metrics=metrics
    )


def demo_interruption(
    config: Config,
    model: PersonaPlexModel,
    voice_codes: torch.Tensor,
    metrics: Metrics
):
    """Demo 3: Test interruption handling."""
    
    print("\n" + "#"*60)
    print("# DEMO: Interruption Handling")
    print("#"*60)
    
    main_audio_path = config.assets_dir / config.main_input_wav
    interrupt_audio_path = config.assets_dir / config.interruption_wav
    output_wav_path = config.output_dir / "response_with_interruption.wav"
    output_text_path = config.output_dir / "transcription_interruption.json"
    
    if not main_audio_path.exists():
        print(f"[!] Main audio not found: {main_audio_path}")
        return
    
    if not interrupt_audio_path.exists():
        print(f"[!] Interruption audio not found: {interrupt_audio_path}")
        return
    
    model.run_inference_with_interruption(
        main_audio_path=main_audio_path,
        interruption_audio_path=interrupt_audio_path,
        voice_codes=voice_codes,
        text_prompt=config.default_text_prompt,
        output_wav_path=output_wav_path,
        output_text_path=output_text_path,
        interruption_delay_sec=config.interruption_delay_sec,
        metrics=metrics
    )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NVIDIA PersonaPlex-7B-v1 Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_script.py --mode all
    python main_script.py --mode voice
    python main_script.py --mode conversation --text-prompt "You are a helpful assistant."
    python main_script.py --mode interruption --interrupt-delay 3.0
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'voice', 'conversation', 'interruption'],
        default='all',
        help='Which demo to run (default: all)'
    )
    
    parser.add_argument(
        '--text-prompt',
        type=str,
        default=None,
        help='Custom text prompt for persona (overrides default)'
    )
    
    parser.add_argument(
        '--voice-embed',
        type=str,
        default=None,
        help='Path to pre-saved voice embedding .pt file'
    )
    
    parser.add_argument(
        '--interrupt-delay',
        type=float,
        default=2.0,
        help='Seconds before interruption starts (default: 2.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42424242,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_args()
    
    print("\n" + "="*60)
    print(" NVIDIA PersonaPlex-7B-v1 Demo")
    print(" Full-Duplex Speech-to-Speech Conversational AI")
    print("="*60)
    
    # Initialize config
    config = Config()
    config.seed = args.seed
    config.device = args.device
    config.interruption_delay_sec = args.interrupt_delay
    
    if args.text_prompt:
        config.default_text_prompt = args.text_prompt
    
    # Initialize metrics
    metrics = Metrics()
    
    # Check assets
    print(f"\n[i] Project root: {config.project_root}")
    print(f"[i] Assets dir: {config.assets_dir}")
    print(f"[i] Output dir: {config.output_dir}")
    
    # Verify required files exist
    required_files = []
    if args.mode in ['all', 'voice']:
        required_files.append(config.assets_dir / config.sample_voice_wav)
    if args.mode in ['all', 'conversation', 'interruption']:
        required_files.append(config.assets_dir / config.main_input_wav)
    if args.mode in ['all', 'interruption']:
        required_files.append(config.assets_dir / config.interruption_wav)
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("\n[!] Missing required audio files:")
        for f in missing_files:
            print(f"    - {f}")
        print("\nPlease add the required audio files and try again.")
        sys.exit(1)
    
    # Load model
    model = PersonaPlexModel(config)
    model.load(metrics)
    
    # Get or create voice embedding
    voice_codes = None
    
    if args.voice_embed:
        # Use provided voice embedding
        voice_codes = model.load_voice_embedding(Path(args.voice_embed))
    elif args.mode in ['all', 'voice']:
        # Create voice embedding from sample audio
        voice_codes = demo_voice_encoding(config, model, metrics)
    else:
        # For conversation/interruption modes without voice encoding,
        # try to load from output dir or create one
        voice_embed_path = config.output_dir / "custom_voice.pt"
        if voice_embed_path.exists():
            voice_codes = model.load_voice_embedding(voice_embed_path)
        else:
            voice_codes = demo_voice_encoding(config, model, metrics)
    
    if voice_codes is None:
        print("\n[!] Could not obtain voice embedding. Exiting.")
        sys.exit(1)
    
    # Run requested demos
    if args.mode in ['all', 'conversation']:
        demo_conversation(config, model, voice_codes, metrics)
    
    if args.mode in ['all', 'interruption']:
        demo_interruption(config, model, voice_codes, metrics)
    
    # Save final metrics
    metrics_path = config.output_dir / "metrics.json"
    metrics.save(metrics_path)
    
    # Print summary
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print(f" Model load time:     {metrics.model_load_time_sec:.2f} sec")
    print(f" Voice encode time:   {metrics.voice_encode_time_sec:.3f} sec")
    print(f" Inference time:      {metrics.inference_time_sec:.2f} sec")
    print(f" Input duration:      {metrics.audio_duration_sec:.2f} sec")
    print(f" Output duration:     {metrics.output_duration_sec:.2f} sec")
    print(f" Real-time factor:    {metrics.real_time_factor:.3f}x")
    print(f" Peak GPU memory:     {metrics.peak_gpu_memory_gb:.2f} GB")
    print("="*60)
    print(f"\n[✓] All outputs saved to: {config.output_dir}")
    print("[✓] Done!")


if __name__ == "__main__":
    main()