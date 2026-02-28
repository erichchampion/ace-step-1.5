#!/usr/bin/env python3
"""
Validate that audio files contain a non-silent, variable waveform.
Used by the generation smoke test to ensure Python and Swift outputs
are real audio (not silence, not constant, no NaN/Inf).

Usage:
    python scripts/validate_audio.py [path1.wav path2.wav ...]

Exit: 0 if every existing file passes; non-zero if any check fails.
Missing files are skipped (not an error). If no paths given, exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_wav_stdlib(path: Path):
    """Load 16-bit PCM WAV using standard library only. Returns (samples, rate)."""
    import struct
    import wave
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    # 16-bit PCM, little-endian
    fmt = "<" + "h" * (len(raw) // 2)
    samples = struct.unpack(fmt, raw)
    return [s / 32768.0 for s in samples], rate


def load_audio(path: Path):
    """Load WAV as float array; support float32 and int16 PCM."""
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32")
        return data.ravel().tolist() if hasattr(data, "ravel") else list(data), rate
    except ImportError:
        pass
    try:
        import scipy.io.wavfile as wav
        rate, data = wav.read(str(path))
        if data.dtype.kind in ("i", "u"):
            data = data.astype("float32") / 32768.0
        else:
            data = data.astype("float32")
        flat = data.ravel() if hasattr(data, "ravel") else data
        return flat.tolist() if hasattr(flat, "tolist") else list(flat), rate
    except ImportError:
        pass
    return _load_wav_stdlib(path)


def _stats(samples: list[float]) -> tuple[float, float]:
    """Return (mean, std) for a list of floats."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n
    return mean, max(0.0, variance) ** 0.5


def validate_one(
    path: Path,
    min_std: float = 0.001,
    min_peak: float = 0.01,
) -> tuple[bool, str]:
    """Check one file: exists, non-empty, variable waveform with real level, no NaN/Inf.
    Rejects flat/silent output (e.g. from a fake decoder that returns zeros).
    """
    if not path.exists():
        return True, "skip (missing)"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        samples, _ = load_audio(path)
    except Exception as e:
        return False, str(e)
    if not samples:
        return False, "no samples"
    for x in samples:
        if x != x or x == float("inf") or x == float("-inf"):
            return False, "NaN or Inf in samples"
    _, std = _stats(samples)
    peak = max(abs(x) for x in samples)
    if std <= min_std:
        return False, f"waveform nearly constant (std={std:.6f}, need >{min_std})"
    if peak < min_peak:
        return False, f"waveform effectively silent (peak={peak:.6f}, need >={min_peak})"
    return True, f"ok (std={std:.6f}, peak={peak:.6f})"


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        return 0
    all_ok = True
    for p in paths:
        ok, msg = validate_one(p)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {p}: {msg}")
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
