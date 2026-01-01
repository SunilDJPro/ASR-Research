#!/usr/bin/env python3
"""
NVIDIA Canary-Qwen-2.5B ASR Evaluation Script
==============================================
Evaluates: RTF (Real-Time Factor), latency, first-token latency, GPU utilization

Requirements:
    pip install --break-system-packages \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" \
        librosa soundfile py3nvml

Usage:
    python canary_asr_eval.py
    
    Then enter audio file paths interactively. Ctrl+C to exit.
"""

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
import numpy as np

# Audio processing
import librosa
import soundfile as sf

# GPU monitoring
from py3nvml import py3nvml

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Evaluation configuration"""
    model_name: str = "nvidia/canary-qwen-2.5b"
    target_sample_rate: int = 16000  # Model expects 16kHz
    max_chunk_duration: float = 30.0  # seconds (model trained on max 40s)
    max_new_tokens: int = 256  # For transcription output
    device: str = "cuda"
    
    # GPU monitoring
    gpu_monitor_interval: float = 0.1  # seconds between GPU samples


# ============================================================================
# GPU Monitor
# ============================================================================

class GPUMonitor:
    """Background thread for GPU utilization monitoring"""
    
    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.device_index = device_index
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Collected metrics
        self.vram_samples: list[float] = []  # MB
        self.util_samples: list[float] = []  # %
        
        # Initialize NVML
        py3nvml.nvmlInit()
        self.handle = py3nvml.nvmlDeviceGetHandleByIndex(device_index)
        
        # Get device info
        self.device_name = py3nvml.nvmlDeviceGetName(self.handle)
        mem_info = py3nvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total_vram_mb = mem_info.total / (1024 ** 2)
    
    def _monitor_loop(self):
        while self._running:
            try:
                mem_info = py3nvml.nvmlDeviceGetMemoryInfo(self.handle)
                util_info = py3nvml.nvmlDeviceGetUtilizationRates(self.handle)
                
                self.vram_samples.append(mem_info.used / (1024 ** 2))
                self.util_samples.append(util_info.gpu)
            except Exception:
                pass
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        self.vram_samples.clear()
        self.util_samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> dict:
        """Stop monitoring and return stats"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        stats = {
            "vram_peak_mb": max(self.vram_samples) if self.vram_samples else 0,
            "vram_avg_mb": np.mean(self.vram_samples) if self.vram_samples else 0,
            "vram_total_mb": self.total_vram_mb,
            "gpu_util_peak_pct": max(self.util_samples) if self.util_samples else 0,
            "gpu_util_avg_pct": np.mean(self.util_samples) if self.util_samples else 0,
            "samples_collected": len(self.vram_samples),
        }
        return stats
    
    def get_current(self) -> tuple[float, float]:
        """Get current VRAM (MB) and utilization (%)"""
        mem_info = py3nvml.nvmlDeviceGetMemoryInfo(self.handle)
        util_info = py3nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return mem_info.used / (1024 ** 2), util_info.gpu
    
    def cleanup(self):
        py3nvml.nvmlShutdown()


# ============================================================================
# Audio Processing
# ============================================================================

def load_and_resample_audio(
    file_path: str, 
    target_sr: int = 16000
) -> tuple[np.ndarray, float, int]:
    """
    Load audio file and resample to target sample rate.
    
    Returns:
        audio: numpy array of audio samples
        duration: audio duration in seconds
        original_sr: original sample rate
    """
    # Load with original sample rate
    audio, original_sr = librosa.load(file_path, sr=None, mono=True)
    duration = len(audio) / original_sr
    
    # Resample if needed
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    return audio, duration, original_sr


def chunk_audio(
    audio: np.ndarray, 
    sample_rate: int, 
    chunk_duration: float
) -> list[np.ndarray]:
    """Split audio into fixed-duration chunks"""
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) > sample_rate * 0.5:  # Skip chunks < 0.5s
            chunks.append(chunk)
    
    return chunks


def save_temp_audio(audio: np.ndarray, sample_rate: int, path: str):
    """Save audio array to wav file"""
    sf.write(path, audio, sample_rate)


# ============================================================================
# Timing Utilities
# ============================================================================

@dataclass
class InferenceMetrics:
    """Metrics from a single inference run"""
    file_path: str
    audio_duration_sec: float
    original_sample_rate: int
    num_chunks: int
    
    # Timing
    preprocessing_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    total_time_sec: float = 0.0
    first_token_latency_sec: float = 0.0  # Time to first output token
    
    # Derived metrics
    rtf: float = 0.0  # Real-Time Factor (inference_time / audio_duration)
    rtfx: float = 0.0  # RTFx = 1/RTF (how many times faster than real-time)
    throughput_sec_per_sec: float = 0.0  # Seconds of audio processed per second
    
    # GPU stats
    gpu_stats: dict = field(default_factory=dict)
    
    # Output
    transcript: str = ""
    
    def compute_derived(self):
        """Compute derived metrics after timing is complete"""
        if self.audio_duration_sec > 0:
            self.rtf = self.inference_time_sec / self.audio_duration_sec
            self.rtfx = 1.0 / self.rtf if self.rtf > 0 else 0
            self.throughput_sec_per_sec = self.audio_duration_sec / self.inference_time_sec if self.inference_time_sec > 0 else 0


class FirstTokenTimer:
    """
    Wrapper to measure first token latency.
    
    Note: NeMo's generate() doesn't expose streaming/token callbacks easily,
    so we measure the time until generate() returns its first chunk for 
    chunked audio, or use a proxy measurement.
    """
    
    def __init__(self):
        self.start_time: float = 0
        self.first_token_time: float = 0
        self.first_token_recorded: bool = False
    
    def start(self):
        self.start_time = time.perf_counter()
        self.first_token_recorded = False
    
    def record_first_token(self):
        if not self.first_token_recorded:
            self.first_token_time = time.perf_counter()
            self.first_token_recorded = True
    
    @property
    def first_token_latency(self) -> float:
        if self.first_token_recorded:
            return self.first_token_time - self.start_time
        return 0.0


# ============================================================================
# ASR Model Wrapper
# ============================================================================

class CanaryASRModel:
    """Wrapper for NVIDIA Canary-Qwen-2.5B model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.temp_dir = Path("/tmp/canary_eval_audio")
        self.temp_dir.mkdir(exist_ok=True)
    
    def load(self):
        """Load the model"""
        print(f"\n{'='*60}")
        print("Loading NVIDIA Canary-Qwen-2.5B...")
        print(f"{'='*60}")
        
        load_start = time.perf_counter()
        
        from nemo.collections.speechlm2.models import SALM
        
        self.model = SALM.from_pretrained(self.config.model_name)
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        load_time = time.perf_counter() - load_start
        
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Device: {self.config.device}")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        
        return load_time
    
    def transcribe_chunk(
        self, 
        audio: np.ndarray, 
        chunk_idx: int,
        first_token_timer: Optional[FirstTokenTimer] = None
    ) -> str:
        """Transcribe a single audio chunk"""
        # Save to temp file (NeMo expects file path)
        temp_path = str(self.temp_dir / f"chunk_{chunk_idx}.wav")
        save_temp_audio(audio, self.config.target_sample_rate, temp_path)
        
        # Build prompt
        prompt = [{
            "role": "user", 
            "content": f"Transcribe the following: {self.model.audio_locator_tag}",
            "audio": [temp_path]
        }]
        
        # Generate
        with torch.no_grad():
            answer_ids = self.model.generate(
                prompts=[prompt],
                max_new_tokens=self.config.max_new_tokens,
            )
        
        # Record first token time (after first chunk completes)
        if first_token_timer and chunk_idx == 0:
            first_token_timer.record_first_token()
        
        # Decode
        transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu().tolist())
        
        # Cleanup temp file
        os.remove(temp_path)
        
        return transcript.strip()
    
    def transcribe(
        self, 
        file_path: str, 
        gpu_monitor: GPUMonitor
    ) -> InferenceMetrics:
        """
        Full transcription pipeline with metrics collection
        """
        metrics = InferenceMetrics(file_path=file_path, audio_duration_sec=0, 
                                   original_sample_rate=0, num_chunks=0)
        
        total_start = time.perf_counter()
        
        # ---- Preprocessing ----
        preproc_start = time.perf_counter()
        
        audio, duration, orig_sr = load_and_resample_audio(
            file_path, 
            self.config.target_sample_rate
        )
        metrics.audio_duration_sec = duration
        metrics.original_sample_rate = orig_sr
        
        # Chunk if needed
        if duration > self.config.max_chunk_duration:
            chunks = chunk_audio(audio, self.config.target_sample_rate, 
                               self.config.max_chunk_duration)
        else:
            chunks = [audio]
        
        metrics.num_chunks = len(chunks)
        metrics.preprocessing_time_sec = time.perf_counter() - preproc_start
        
        # ---- Inference ----
        first_token_timer = FirstTokenTimer()
        transcripts = []
        
        # Start GPU monitoring
        gpu_monitor.start()
        
        inference_start = time.perf_counter()
        first_token_timer.start()
        
        # Warm up CUDA if first run
        torch.cuda.synchronize()
        
        for i, chunk in enumerate(chunks):
            transcript = self.transcribe_chunk(chunk, i, first_token_timer)
            transcripts.append(transcript)
        
        torch.cuda.synchronize()
        metrics.inference_time_sec = time.perf_counter() - inference_start
        
        # Stop GPU monitoring
        metrics.gpu_stats = gpu_monitor.stop()
        
        # ---- Finalize ----
        metrics.first_token_latency_sec = first_token_timer.first_token_latency
        metrics.transcript = " ".join(transcripts)
        metrics.total_time_sec = time.perf_counter() - total_start
        metrics.compute_derived()
        
        return metrics
    
    def warmup(self, gpu_monitor: GPUMonitor):
        """Run a warmup inference to initialize CUDA kernels"""
        print("\nRunning warmup inference...")
        
        # Generate 2 seconds of silence
        silence = np.zeros(int(2 * self.config.target_sample_rate), dtype=np.float32)
        # Add tiny noise to avoid edge cases
        silence += np.random.randn(*silence.shape) * 1e-6
        
        temp_path = str(self.temp_dir / "warmup.wav")
        save_temp_audio(silence, self.config.target_sample_rate, temp_path)
        
        gpu_monitor.start()
        
        with torch.no_grad():
            prompt = [{
                "role": "user",
                "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                "audio": [temp_path]
            }]
            _ = self.model.generate(prompts=[prompt], max_new_tokens=16)
        
        torch.cuda.synchronize()
        warmup_stats = gpu_monitor.stop()
        
        os.remove(temp_path)
        
        print(f"Warmup complete. Peak VRAM: {warmup_stats['vram_peak_mb']:.0f} MB")


# ============================================================================
# Display Utilities
# ============================================================================

def print_metrics(metrics: InferenceMetrics):
    """Pretty print inference metrics"""
    print(f"\n{'─'*60}")
    print(f"FILE: {metrics.file_path}")
    print(f"{'─'*60}")
    
    print(f"\n📊 AUDIO INFO:")
    print(f"   Duration:        {metrics.audio_duration_sec:.2f}s")
    print(f"   Original SR:     {metrics.original_sample_rate} Hz")
    print(f"   Chunks:          {metrics.num_chunks}")
    
    print(f"\n⏱️  TIMING:")
    print(f"   Preprocessing:   {metrics.preprocessing_time_sec*1000:.1f} ms")
    print(f"   Inference:       {metrics.inference_time_sec*1000:.1f} ms")
    print(f"   Total:           {metrics.total_time_sec*1000:.1f} ms")
    print(f"   First Token:     {metrics.first_token_latency_sec*1000:.1f} ms")
    
    print(f"\n🚀 PERFORMANCE:")
    print(f"   RTF:             {metrics.rtf:.4f} (lower is better)")
    print(f"   RTFx:            {metrics.rtfx:.1f}x faster than real-time")
    print(f"   Throughput:      {metrics.throughput_sec_per_sec:.2f} sec audio/sec")
    
    gpu = metrics.gpu_stats
    if gpu:
        print(f"\n🖥️  GPU UTILIZATION:")
        print(f"   VRAM Peak:       {gpu['vram_peak_mb']:.0f} / {gpu['vram_total_mb']:.0f} MB "
              f"({100*gpu['vram_peak_mb']/gpu['vram_total_mb']:.1f}%)")
        print(f"   VRAM Avg:        {gpu['vram_avg_mb']:.0f} MB")
        print(f"   GPU Util Peak:   {gpu['gpu_util_peak_pct']:.0f}%")
        print(f"   GPU Util Avg:    {gpu['gpu_util_avg_pct']:.1f}%")
    
    print(f"\n📝 TRANSCRIPT:")
    # Wrap transcript for readability
    transcript = metrics.transcript
    max_width = 56
    words = transcript.split()
    lines = []
    current_line = "   "
    for word in words:
        if len(current_line) + len(word) + 1 > max_width:
            lines.append(current_line)
            current_line = "   " + word
        else:
            current_line += " " + word if current_line != "   " else word
    if current_line.strip():
        lines.append(current_line)
    
    for line in lines[:10]:  # Limit display
        print(line)
    if len(lines) > 10:
        print(f"   ... ({len(lines) - 10} more lines)")
    
    print(f"{'─'*60}\n")


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     NVIDIA Canary-Qwen-2.5B ASR Evaluation Playground        ║
║                                                              ║
║  Metrics: RTF, Latency, First-Token Latency, GPU Util        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ============================================================================
# Main
# ============================================================================

def main():
    print_banner()
    
    config = Config()
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_index=0, interval=config.gpu_monitor_interval)
    print(f"GPU Detected: {gpu_monitor.device_name}")
    print(f"Total VRAM: {gpu_monitor.total_vram_mb:.0f} MB")
    
    current_vram, current_util = gpu_monitor.get_current()
    print(f"Current VRAM Usage: {current_vram:.0f} MB")
    
    # Load model
    model = CanaryASRModel(config)
    model.load()
    
    # Warmup
    model.warmup(gpu_monitor)
    
    # Interactive loop
    print("\n" + "="*60)
    print("Ready for inference!")
    print("Enter audio file path (.wav) or 'q' to quit")
    print("="*60)
    
    session_metrics: list[InferenceMetrics] = []
    
    try:
        while True:
            print()
            file_path = input("🎤 Audio file path: ").strip()
            
            if file_path.lower() in ('q', 'quit', 'exit'):
                break
            
            if not file_path:
                continue
            
            # Handle quoted paths
            file_path = file_path.strip("'\"")
            
            # Validate file
            if not os.path.isfile(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            if not file_path.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
                print("⚠️  Warning: Expected .wav file, attempting anyway...")
            
            try:
                metrics = model.transcribe(file_path, gpu_monitor)
                session_metrics.append(metrics)
                print_metrics(metrics)
                
            except torch.cuda.OutOfMemoryError:
                print("❌ CUDA Out of Memory! Try a shorter audio file or use A6000.")
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        # Print session summary
        if session_metrics:
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Files processed: {len(session_metrics)}")
            
            total_audio = sum(m.audio_duration_sec for m in session_metrics)
            total_inference = sum(m.inference_time_sec for m in session_metrics)
            avg_rtfx = np.mean([m.rtfx for m in session_metrics])
            avg_first_token = np.mean([m.first_token_latency_sec for m in session_metrics])
            
            print(f"Total audio:     {total_audio:.1f}s")
            print(f"Total inference: {total_inference:.1f}s")
            print(f"Average RTFx:    {avg_rtfx:.1f}x")
            print(f"Avg First Token: {avg_first_token*1000:.1f}ms")
            print("="*60)
        
        # Cleanup
        gpu_monitor.cleanup()
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()