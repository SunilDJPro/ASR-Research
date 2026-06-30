"""
parakeet_stream.py — a thin ctypes wrapper over parakeet.cpp's streaming C-API,
targeting the cache-aware EOU model (parakeet_realtime_eou_120m-v1).

Covers two input paths:
  1. WAV files  -> read with the stdlib `wave` module, convert to 16 kHz mono f32,
                   feed in chunks (simulates a live stream over a recorded clip).
  2. In-memory PCM (e.g. from a mic via sounddevice/pyaudio) -> feed raw f32 frames
                   as they arrive.

This wraps libparakeet.so. Build it first:
    cmake -B build-shared -DPARAKEET_SHARED=ON -DPARAKEET_BUILD_CLI=ON
    cmake --build build-shared -j
and point PARAKEET_LIB at build-shared/libparakeet.so (or pass lib_path=...).

IMPORTANT — verify signatures against your build before trusting this:
    nm -D build-shared/libparakeet.so | grep parakeet_capi
The signatures below follow the documented C-API and the ABI v5 note that
`*eou_out` from stream_feed is an EVENT BITMASK, not a plain bool. If your
ABI differs, adjust EouEvent / the bit decoding accordingly.

Streaming requires 16 kHz, mono, float32 PCM in [-1, 1].
"""

import ctypes
import os
import wave
import audioop
from ctypes import (
    c_char_p, c_void_p, c_int, c_float, c_size_t, POINTER, byref,
)

# ---- ABI v5 event bitmask (verify against your header) ----------------------
# *eou_out is a bitmask. Bit 0 = end-of-utterance, bit 1 = backchannel (EOB).
EVENT_EOU = 0x1
EVENT_EOB = 0x2

TARGET_SR = 16000


class ParakeetError(RuntimeError):
    pass


def _default_lib_path() -> str:
    return os.environ.get("PARAKEET_LIB", "libparakeet.so")


class _Lib:
    """Loads libparakeet.so and declares the C-API signatures once."""

    def __init__(self, lib_path: str):
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise ParakeetError(
                f"could not load libparakeet from {lib_path!r}: {e}\n"
                "Build with -DPARAKEET_SHARED=ON and set PARAKEET_LIB."
            ) from e
        self._declare()

    def _declare(self):
        L = self.lib

        # --- lifecycle ---
        L.parakeet_capi_load.argtypes = [c_char_p]
        L.parakeet_capi_load.restype = c_void_p

        L.parakeet_capi_free.argtypes = [c_void_p]
        L.parakeet_capi_free.restype = None

        L.parakeet_capi_last_error.argtypes = [c_void_p]
        L.parakeet_capi_last_error.restype = c_char_p

        L.parakeet_capi_free_string.argtypes = [c_void_p]
        L.parakeet_capi_free_string.restype = None

        # --- streaming (cache-aware EOU model) ---
        L.parakeet_capi_stream_begin.argtypes = [c_void_p]
        L.parakeet_capi_stream_begin.restype = c_void_p

        # parakeet_capi_stream_feed(stream, const float* pcm, size_t n,
        #                           int* eou_out) -> char* (newly finalized text)
        L.parakeet_capi_stream_feed.argtypes = [
            c_void_p, POINTER(c_float), c_size_t, POINTER(c_int),
        ]
        L.parakeet_capi_stream_feed.restype = c_void_p  # malloc'd char*; free it

        L.parakeet_capi_stream_finalize.argtypes = [c_void_p]
        L.parakeet_capi_stream_finalize.restype = c_void_p  # malloc'd char*

        L.parakeet_capi_stream_free.argtypes = [c_void_p]
        L.parakeet_capi_stream_free.restype = None

    def take_string(self, ptr) -> str:
        """Decode a malloc'd char* from the C-API and free it. NULL -> ''."""
        if not ptr:
            return ""
        try:
            return ctypes.cast(ptr, c_char_p).value.decode("utf-8", "replace")
        finally:
            self.lib.parakeet_capi_free_string(ptr)


class EouEvent:
    """Decoded event bitmask from a feed() call."""

    __slots__ = ("eou", "eob", "raw")

    def __init__(self, raw: int):
        self.raw = raw
        self.eou = bool(raw & EVENT_EOU)
        self.eob = bool(raw & EVENT_EOB)

    def __bool__(self):
        return self.raw != 0

    def __repr__(self):
        return f"EouEvent(eou={self.eou}, eob={self.eob}, raw={self.raw})"


class StreamSession:
    """A single live streaming session. Feed f32 PCM chunks, get finalized text."""

    def __init__(self, lib: _Lib, ctx: c_void_p):
        self._lib = lib
        self._stream = lib.lib.parakeet_capi_stream_begin(ctx)
        if not self._stream:
            raise ParakeetError("parakeet_capi_stream_begin returned NULL")
        self._closed = False

    def feed(self, pcm_f32) -> tuple[str, EouEvent]:
        """Feed a chunk of 16 kHz mono float32 PCM.

        pcm_f32: a ctypes float array, a bytes/bytearray of raw f32, or any
                 buffer convertible via (c_float * n). Returns (new_text, event).
        """
        if self._closed:
            raise ParakeetError("session already finalized")

        arr, n = _as_float_array(pcm_f32)
        eou = c_int(0)
        ptr = self._lib.lib.parakeet_capi_stream_feed(
            self._stream, arr, c_size_t(n), byref(eou)
        )
        text = self._lib.take_string(ptr)
        return text, EouEvent(eou.value)

    def finalize(self) -> str:
        """Flush the end-of-stream tail. Idempotent-safe; frees the session."""
        if self._closed:
            return ""
        ptr = self._lib.lib.parakeet_capi_stream_finalize(self._stream)
        tail = self._lib.take_string(ptr)
        self._lib.lib.parakeet_capi_stream_free(self._stream)
        self._stream = None
        self._closed = True
        return tail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalize()


class Parakeet:
    """Loads a GGUF model once; spawn StreamSessions from it."""

    def __init__(self, model_path: str, lib_path: str | None = None):
        self._lib = _Lib(lib_path or _default_lib_path())
        self._ctx = self._lib.lib.parakeet_capi_load(model_path.encode("utf-8"))
        if not self._ctx:
            err = self._lib.lib.parakeet_capi_last_error(None)
            msg = err.decode("utf-8", "replace") if err else "unknown error"
            raise ParakeetError(f"failed to load model {model_path!r}: {msg}")

    def stream(self) -> StreamSession:
        return StreamSession(self._lib, self._ctx)

    def close(self):
        if self._ctx:
            self._lib.lib.parakeet_capi_free(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- helpers ----------------------------------------------------------------

def _as_float_array(pcm) -> tuple["ctypes.Array", int]:
    """Coerce input into a (c_float * n) array. Accepts an existing c_float
    array (passed through), bytes/bytearray of raw little-endian f32, or any
    Python sequence of floats."""
    if isinstance(pcm, ctypes.Array) and pcm._type_ is c_float:
        return pcm, len(pcm)
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        n = len(pcm) // 4
        arr = (c_float * n).from_buffer_copy(bytes(pcm))
        return arr, n
    # assume an iterable of floats
    seq = list(pcm)
    arr = (c_float * len(seq))(*seq)
    return arr, len(seq)


def wav_to_f32_16k_mono(path: str) -> list[float]:
    """Read a WAV file and return 16 kHz mono float32 samples in [-1, 1].

    Uses stdlib wave + audioop only (no numpy dependency). Handles 8/16/32-bit
    PCM, downmixes stereo, and resamples to 16 kHz.
    """
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if n_ch > 1:
        raw = audioop.tomono(raw, sampwidth, 0.5, 0.5) if n_ch == 2 else _downmix(raw, sampwidth, n_ch)

    if sr != TARGET_SR:
        raw, _ = audioop.ratecv(raw, sampwidth, 1, sr, TARGET_SR, None)

    # normalize to float32 in [-1, 1]
    max_val = float(1 << (8 * sampwidth - 1))
    if sampwidth == 1:
        # wave gives unsigned 8-bit; center it
        return [(b - 128) / 128.0 for b in raw]
    fmt_width = sampwidth
    samples = _unpack_pcm(raw, fmt_width)
    return [s / max_val for s in samples]


def _downmix(raw, sampwidth, n_ch):
    # generic downmix for >2 channels: average channels
    frames = len(raw) // (sampwidth * n_ch)
    out = bytearray()
    for i in range(frames):
        acc = 0
        for c in range(n_ch):
            off = (i * n_ch + c) * sampwidth
            acc += int.from_bytes(raw[off:off + sampwidth], "little", signed=True)
        avg = acc // n_ch
        out += avg.to_bytes(sampwidth, "little", signed=True)
    return bytes(out)


def _unpack_pcm(raw, sampwidth):
    out = []
    for off in range(0, len(raw), sampwidth):
        out.append(int.from_bytes(raw[off:off + sampwidth], "little", signed=True))
    return out


def stream_wav(model: Parakeet, wav_path: str, chunk_ms: int = 160):
    """Stream a recorded WAV through the model in chunk_ms slices, printing
    finalized text and EOU/EOB markers as they arrive. Returns the full
    transcript. (The EOU model requires >=160ms of audio to start.)"""
    samples = wav_to_f32_16k_mono(wav_path)
    chunk = max(1, int(TARGET_SR * chunk_ms / 1000))
    pieces = []
    with model.stream() as s:
        for i in range(0, len(samples), chunk):
            block = samples[i:i + chunk]
            arr = (c_float * len(block))(*block)
            text, ev = s.feed(arr)
            if text:
                pieces.append(text)
                print(text, end="", flush=True)
            if ev.eou:
                print(" [EOU]", end="", flush=True)
            if ev.eob:
                print(" [EOB]", end="", flush=True)
        tail = s.finalize()
        if tail:
            pieces.append(tail)
            print(tail)
    return "".join(pieces)


def stream_mic(model: Parakeet, device=None, blocksize: int = 2560):
    """Live microphone streaming. Requires `sounddevice` (and numpy).

    Opens the default (or given) input device at 16 kHz mono float32 and feeds
    frames into a StreamSession as they arrive, printing finalized text and
    EOU/EOB markers. Ctrl-C to stop; the tail is flushed on exit.

    device:    sounddevice device index/name, or None for the default input.
    blocksize: frames per callback (2560 = 160 ms at 16 kHz, the model's chunk).
    """
    import queue
    import sounddevice as sd  # imported lazily so file-only use needs no deps

    q: "queue.Queue[bytes]" = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}", flush=True)
        # indata is float32 (channels=1) -> raw f32 bytes
        q.put(bytes(indata))

    print("Listening (Ctrl-C to stop)... speak into the mic.")
    with model.stream() as s:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                            blocksize=blocksize, device=device, callback=cb):
            try:
                while True:
                    pcm_bytes = q.get()           # raw little-endian f32
                    text, ev = s.feed(pcm_bytes)  # bytes path handled by wrapper
                    if text:
                        print(text, end="", flush=True)
                    if ev.eou:
                        print(" [EOU]", end="", flush=True)
                    if ev.eob:
                        print(" [EOB]", end="", flush=True)
            except KeyboardInterrupt:
                pass
        tail = s.finalize()
        if tail:
            print(tail)
    print("\n[stopped]")


if __name__ == "__main__":
    import sys

    # usage:
    #   replay a wav:  python parakeet_stream.py <eou_model.gguf> <audio.wav>
    #   live mic:      python parakeet_stream.py <eou_model.gguf> --mic [device]
    if len(sys.argv) < 3:
        print("usage:")
        print("  PARAKEET_LIB=.../libparakeet.so \\")
        print("    python parakeet_stream.py <eou_model.gguf> <audio.wav>")
        print("  PARAKEET_LIB=.../libparakeet.so \\")
        print("    python parakeet_stream.py <eou_model.gguf> --mic [device]")
        sys.exit(1)

    model_path, arg2 = sys.argv[1], sys.argv[2]
    with Parakeet(model_path) as pk:
        if arg2 == "--mic":
            device = sys.argv[3] if len(sys.argv) > 3 else None
            # sounddevice accepts an int index or a name substring
            if device is not None and device.isdigit():
                device = int(device)
            stream_mic(pk, device=device)
        else:
            print("--- streaming transcript ---")
            full = stream_wav(pk, arg2)
            print("\n--- full ---")
            print(full)