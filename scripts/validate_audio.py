#!/usr/bin/env python3
"""
Validate that audio files contain a non-silent, variable waveform.
Used by the generation smoke test to ensure Python and Swift outputs
are real audio (not silence, not constant, no NaN/Inf).

Usage:
    python scripts/validate_audio.py [path1.wav path2.wav ...]
    python scripts/validate_audio.py --expected-duration 5.0 path1.wav path2.wav
    python scripts/validate_audio.py --compare path1.wav path2.wav

Exit: 0 if every existing file passes; non-zero if any check fails.
Missing files are skipped (not an error). If no paths given, exits 0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _load_wav_stdlib(path: Path):
    """Load 16-bit PCM WAV using standard library only. Returns (samples, rate, channels)."""
    import struct
    import wave
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        channels = w.getnchannels()
        raw = w.readframes(n)
    # 16-bit PCM, little-endian
    fmt = "<" + "h" * (len(raw) // 2)
    samples = struct.unpack(fmt, raw)
    return [s / 32768.0 for s in samples], rate, channels


def load_audio(path: Path):
    """Load WAV as float array; support float32 and int16 PCM. Returns (samples, rate, channels)."""
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32")
        channels = 1 if data.ndim == 1 else data.shape[1]
        return data.ravel().tolist() if hasattr(data, "ravel") else list(data), rate, channels
    except ImportError:
        pass
    try:
        import scipy.io.wavfile as wav
        rate, data = wav.read(str(path))
        if data.dtype.kind in ("i", "u"):
            data = data.astype("float32") / 32768.0
        else:
            data = data.astype("float32")
        channels = 1 if data.ndim == 1 else data.shape[1]
        flat = data.ravel() if hasattr(data, "ravel") else data
        return flat.tolist() if hasattr(flat, "tolist") else list(flat), rate, channels
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


def _channel_stats(samples: list[float], channels: int) -> list[tuple[float, float]]:
    """Return per-channel (mean, std) for interleaved samples."""
    if channels <= 1:
        return [_stats(samples)]
    # Channel-interleaved: [L0, R0, L1, R1, ...]
    result = []
    for ch in range(channels):
        ch_samples = samples[ch::channels]
        result.append(_stats(ch_samples))
    return result


def _spectral_flatness(samples: list[float], channels: int = 1) -> float:
    """Approximate spectral flatness (0 = tonal, 1 = noise-like).
    Higher values indicate more even spectral distribution.
    Uses a simple FFT-based approach; returns -1 if numpy is unavailable."""
    try:
        import numpy as np
    except ImportError:
        return -1.0
    arr = np.array(samples, dtype=np.float32)
    if channels > 1:
        # Take first channel for spectral analysis
        arr = arr[::channels]
    if len(arr) < 256:
        return -1.0
    fft = np.abs(np.fft.rfft(arr[:min(len(arr), 65536)]))
    fft = fft[1:]  # skip DC
    fft = fft + 1e-10  # avoid log(0)
    geometric_mean = np.exp(np.mean(np.log(fft)))
    arithmetic_mean = np.mean(fft)
    if arithmetic_mean < 1e-10:
        return 0.0
    return float(geometric_mean / arithmetic_mean)


def validate_one(
    path: Path,
    min_std: float = 0.001,
    min_peak: float = 0.01,
    expected_duration: float | None = None,
    duration_tolerance: float = 0.5,
) -> tuple[bool, str]:
    """Check one file: exists, non-empty, variable waveform with real level, no NaN/Inf.
    Optionally check duration and stereo quality.
    Rejects flat/silent output (e.g. from a fake decoder that returns zeros).
    """
    if not path.exists():
        return True, "skip (missing)"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        samples, rate, channels = load_audio(path)
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

    # Duration check
    total_samples = len(samples) // max(channels, 1)
    actual_duration = total_samples / rate if rate > 0 else 0
    duration_info = f"duration={actual_duration:.2f}s"
    if expected_duration is not None and expected_duration > 0:
        if abs(actual_duration - expected_duration) > duration_tolerance:
            return False, (
                f"duration mismatch: expected ~{expected_duration:.1f}s, "
                f"got {actual_duration:.2f}s (tolerance ±{duration_tolerance}s)"
            )

    # Per-channel statistics
    ch_stats = _channel_stats(samples, channels)
    ch_info_parts = []
    for i, (cm, cs) in enumerate(ch_stats):
        label = f"ch{i}" if channels > 1 else "mono"
        ch_info_parts.append(f"{label}(mean={cm:.4f},std={cs:.4f})")
    ch_info = " ".join(ch_info_parts)

    # Stereo independence check: for stereo, verify channels aren't identical
    stereo_info = ""
    if channels == 2 and len(samples) >= 4:
        left = samples[0::2]
        right = samples[1::2]
        # Check if channels are independent (not identical)
        n_check = min(len(left), len(right), 10000)
        diff_sum = sum(abs(left[i] - right[i]) for i in range(n_check))
        avg_diff = diff_sum / n_check if n_check > 0 else 0
        if avg_diff < 1e-6:
            stereo_info = " WARN:channels_identical"
        else:
            stereo_info = f" stereo_diff={avg_diff:.4f}"

    # Spectral flatness
    sf = _spectral_flatness(samples, channels)
    spectral_info = f" spectral_flatness={sf:.3f}" if sf >= 0 else ""

    msg = (
        f"ok (std={std:.6f}, peak={peak:.6f}, {duration_info}, "
        f"rate={rate}, ch={channels}, {ch_info}{stereo_info}{spectral_info})"
    )
    return True, msg


def compare_files(paths: list[Path]) -> list[str]:
    """Compare basic statistics between multiple output files."""
    if len(paths) < 2:
        return []
    results: dict[str, tuple[float, float, float]] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            samples, rate, channels = load_audio(p)
            mean, std = _stats(samples)
            peak = max(abs(x) for x in samples) if samples else 0
            results[p.name] = (mean, std, peak)
        except Exception:
            continue
    if len(results) < 2:
        return []
    msgs = ["Cross-file comparison:"]
    names = list(results.keys())
    for i, name in enumerate(names):
        m, s, p = results[name]
        msgs.append(f"  {name}: mean={m:.6f} std={s:.6f} peak={p:.6f}")
    # Compare first two
    if len(names) >= 2:
        (m1, s1, p1), (m2, s2, p2) = results[names[0]], results[names[1]]
        mean_diff = abs(m1 - m2)
        std_ratio = s1 / s2 if s2 > 0 else float("inf")
        peak_ratio = p1 / p2 if p2 > 0 else float("inf")
        msgs.append(
            f"  {names[0]} vs {names[1]}: "
            f"mean_diff={mean_diff:.6f}, std_ratio={std_ratio:.3f}, peak_ratio={peak_ratio:.3f}"
        )
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate audio files from smoke test")
    parser.add_argument("paths", nargs="*", type=Path, help="WAV files to validate")
    parser.add_argument(
        "--expected-duration", type=float, default=None,
        help="Expected audio duration in seconds (tolerance ±0.5s)"
    )
    parser.add_argument(
        "--duration-tolerance", type=float, default=0.5,
        help="Tolerance for duration check in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare statistics across all provided files"
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        return 0
    all_ok = True
    for p in paths:
        ok, msg = validate_one(
            p,
            expected_duration=args.expected_duration,
            duration_tolerance=args.duration_tolerance,
        )
        status = "PASS" if ok else "FAIL"
        print(f"{status} {p}: {msg}")
        if not ok:
            all_ok = False

    if args.compare:
        for line in compare_files(paths):
            print(line)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
