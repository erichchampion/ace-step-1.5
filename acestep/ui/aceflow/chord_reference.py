"""Utilities for parsing chord symbols and rendering lightweight harmonic reference WAV files.

This module keeps the AceFlow chord-reference workflow self-contained: it parses
common chord symbols, chooses compact voicings with simple continuity heuristics,
and synthesizes a mono WAV preview used for harmonic conditioning.
"""

from __future__ import annotations

import io
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

NOTE_INDEX = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11,
}

_DESCRIPTOR_ALIASES = {
    '': 'maj',
    'maj': 'maj',
    'major': 'maj',
    'm': 'min',
    'min': 'min',
    'minor': 'min',
    'maj7': 'maj7',
    'm7': 'min7',
    'min7': 'min7',
    '7': 'dom7',
    'dom7': 'dom7',
    'dim': 'dim',
    'dim7': 'dim7',
    'aug': 'aug',
    '+': 'aug',
    'sus': 'sus4',
    'sus4': 'sus4',
    'sus2': 'sus2',
    'add9': 'add9',
    '6': '6',
    'm6': 'min6',
    '9': '9',
    'maj9': 'maj9',
    'm9': 'min9',
    '7#5': '7#5',
}

_BASE_INTERVALS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    'dim': [0, 3, 6],
    'dim7': [0, 3, 6, 9],
    'aug': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'add9': [0, 4, 7, 14],
    '6': [0, 4, 7, 9],
    'min6': [0, 3, 7, 9],
    '9': [0, 4, 7, 10, 14],
    'maj9': [0, 4, 7, 11, 14],
    'min9': [0, 3, 7, 10, 14],
    '7#5': [0, 4, 8, 10],
}

MAX_RENDER_DURATION_SEC = 600.0


@dataclass
class ParsedChord:
    """Normalized representation of a chord symbol used by the preview renderer.

    Attributes:
        original: Original chord token received from the UI or section map.
        normalized: Sanitized symbol used for parsing and debug metadata.
        root_pc: Root pitch class in semitones where C=0.
        chord_pcs: Distinct pitch classes included in the resolved chord.
        bass_pc: Pitch class used for the bass note after slash-chord parsing.
        descriptor: Canonical internal descriptor such as ``maj7`` or ``min9``.
        warning: Optional fallback marker when the descriptor needed heuristics.
    """

    original: str
    normalized: str
    root_pc: int
    chord_pcs: list[int]
    bass_pc: int
    descriptor: str
    warning: Optional[str] = None


def midi_to_freq(midi: float) -> float:
    """Convert a MIDI note number to frequency in hertz.

    Args:
        midi: MIDI note value, including fractional notes if needed.

    Returns:
        float: Frequency in hertz referenced to A4 = 440 Hz.
    """
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or '').strip()
    s = s.replace('♯', '#').replace('♭', 'b').replace('Δ', 'maj')
    return re.sub(r'\s+', '', s)


def parse_chord_symbol(symbol: str) -> Optional[ParsedChord]:
    """Parse a chord symbol into a normalized structure for voicing and rendering.

    Args:
        symbol: Raw chord symbol such as ``Cmaj7``, ``Am7``, or ``F/G``.

    Returns:
        Optional[ParsedChord]: Parsed chord data, or ``None`` when the input
        cannot be interpreted by the AceFlow chord parser.

    Notes:
        Uppercase major aliases like ``CM7`` and ``FM9`` are handled before the
        lowercase fallback heuristics to avoid accidental minor resolution.
    """
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return None
    m = re.match(r'^([A-G](?:#|b)?)([^/]*)?(?:/([A-G](?:#|b)?))?$', normalized)
    if not m:
        return None
    root_name = m.group(1)
    raw_desc = (m.group(2) or '').strip()
    bass_name = (m.group(3) or '').strip()
    root_pc = NOTE_INDEX.get(root_name)
    if root_pc is None:
        return None
    bass_pc = NOTE_INDEX.get(bass_name, root_pc) if bass_name else root_pc
    desc = raw_desc.strip()
    descriptor = _DESCRIPTOR_ALIASES.get(desc)
    warning = None
    uppercase_major_aliases = {
        'M': 'maj',
        'M6': '6',
        'M7': 'maj7',
        'M9': 'maj9',
        'M11': 'maj9',
        'M13': 'maj9',
    }
    uppercase_major_descriptor = uppercase_major_aliases.get(desc)
    if descriptor is None and uppercase_major_descriptor is not None:
        descriptor = uppercase_major_descriptor
        warning = 'descriptor_fallback'
    if descriptor is None:
        lc = desc.lower()
        if 'maj9' in lc:
            descriptor = 'maj9'
        elif 'maj7' in lc:
            descriptor = 'maj7'
            warning = 'descriptor_fallback' if lc != 'maj7' else None
        elif lc in {'m7b5', 'ø', 'ø7'}:
            descriptor = 'dim'
            warning = 'descriptor_fallback'
        elif 'm9' in lc or 'min9' in lc:
            descriptor = 'min9'
            warning = 'descriptor_fallback' if lc not in {'m9', 'min9'} else None
        elif '13' in lc or '11' in lc:
            if 'maj' in lc:
                descriptor = 'maj9'
            elif lc.startswith('m') or 'min' in lc:
                descriptor = 'min9'
            else:
                descriptor = '9'
            warning = 'descriptor_fallback'
        elif lc in {'9'} or lc.startswith('9'):
            descriptor = '9'
            warning = 'descriptor_fallback' if lc != '9' else None
        elif 'm7' in lc or 'min7' in lc:
            descriptor = 'min7'
            warning = 'descriptor_fallback' if lc not in {'m7', 'min7'} else None
        elif 'add9' in lc:
            descriptor = 'add9'
            warning = 'descriptor_fallback' if lc != 'add9' else None
        elif 'sus2' in lc:
            descriptor = 'sus2'
            warning = 'descriptor_fallback' if lc != 'sus2' else None
        elif 'sus' in lc:
            descriptor = 'sus4'
            warning = 'descriptor_fallback' if lc != 'sus' and lc != 'sus4' else None
        elif 'dim7' in lc:
            descriptor = 'dim7'
            warning = 'descriptor_fallback' if lc != 'dim7' else None
        elif 'dim' in lc:
            descriptor = 'dim'
            warning = 'descriptor_fallback' if lc != 'dim' else None
        elif '7#5' in lc:
            descriptor = '7#5'
            warning = 'descriptor_fallback' if lc != '7#5' else None
        elif 'aug' in lc or lc == '+':
            descriptor = 'aug'
            warning = 'descriptor_fallback' if lc not in {'aug', '+'} else None
        elif lc == '7' or lc.startswith('7'):
            descriptor = 'dom7'
            warning = 'descriptor_fallback' if lc != '7' else None
        elif lc.startswith('m') or lc.startswith('min'):
            descriptor = 'min'
            warning = 'descriptor_fallback' if lc not in {'m', 'min', 'minor'} else None
        else:
            descriptor = 'maj'
            warning = 'descriptor_fallback' if lc else None
    intervals = _BASE_INTERVALS[descriptor]
    chord_pcs = sorted({(root_pc + interval) % 12 for interval in intervals})
    return ParsedChord(
        original=str(symbol or ''),
        normalized=normalized,
        root_pc=root_pc,
        chord_pcs=chord_pcs,
        bass_pc=bass_pc,
        descriptor=descriptor,
        warning=warning,
    )


def _warning_debug_entry(symbol: str, parsed: Optional[ParsedChord], reason: str, fallback: str) -> dict:
    entry = {
        'symbol': str(symbol or ''),
        'reason': str(reason or ''),
        'fallback': str(fallback or ''),
    }
    if parsed is not None:
        entry['normalized_input'] = parsed.normalized
        entry['resolved_descriptor'] = parsed.descriptor
    return entry


def _pitch_candidates(pc: int, lo: int, hi: int) -> list[int]:
    return [midi for midi in range(lo, hi + 1) if midi % 12 == pc]


def _best_bass_midi(pc: int, previous: Optional[int]) -> int:
    candidates = _pitch_candidates(pc, 36, 55)
    target = previous if previous is not None else 43
    return min(candidates, key=lambda x: (abs(x - target), abs(x - 43)))


def _compact_voicing(root_midi: int, intervals: list[int]) -> list[int]:
    voiced: list[int] = []
    for idx, interval in enumerate(intervals):
        midi = root_midi + interval
        while midi < 55:
            midi += 12
        while midi > 76:
            midi -= 12
        while idx > 0 and midi <= voiced[-1]:
            midi += 12
        voiced.append(midi)
    for i in range(1, len(voiced)):
        while voiced[i] - voiced[0] > 14 and voiced[i] - 12 > voiced[i - 1]:
            voiced[i] -= 12
    voiced = sorted(voiced)
    return [min(79, max(54, v)) for v in voiced]


def choose_voicing(chord: ParsedChord, previous_pad: Optional[list[int]], previous_bass: Optional[int]) -> tuple[int, list[int]]:
    """Choose a bass note and compact pad voicing for a parsed chord.

    Args:
        chord: Parsed chord returned by :func:`parse_chord_symbol`.
        previous_pad: Previously rendered pad voicing used for continuity.
        previous_bass: Previously rendered bass MIDI note used to reduce leaps.

    Returns:
        tuple[int, list[int]]: Bass MIDI note and ordered pad MIDI notes.
    """
    bass_midi = _best_bass_midi(chord.bass_pc, previous_bass)
    root_pc = chord.root_pc
    intervals = sorted({(pc - root_pc) % 12 for pc in chord.chord_pcs})
    candidate_roots = _pitch_candidates(root_pc, 55, 67) or [60]
    prev_pad = previous_pad or [60, 64, 67]
    prev_center = sum(prev_pad) / len(prev_pad)
    best_pad = None
    best_score = None
    for root_midi in candidate_roots:
        for inversion in range(len(intervals)):
            rotated = intervals[inversion:] + [x + 12 for x in intervals[:inversion]]
            pad = _compact_voicing(root_midi, rotated)
            if len(pad) >= 5:
                pad = [pad[0], pad[1], pad[2], pad[-1]]
            if len(pad) < 3:
                continue
            center = sum(pad) / len(pad)
            spread = pad[-1] - pad[0]
            leap = sum(abs(m - prev_pad[min(i, len(prev_pad) - 1)]) for i, m in enumerate(pad))
            score = abs(center - 64.5) + abs(center - prev_center) * 0.8 + spread * 0.35 + leap * 0.55
            if best_score is None or score < best_score:
                best_score = score
                best_pad = pad
    return bass_midi, (best_pad or [60, 64, 67])


def _envelope(length: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float32)
    attack_n = max(1, int(sr * max(0.001, attack)))
    decay_n = max(1, int(sr * max(0.001, decay)))
    release_n = max(1, int(sr * max(0.001, release)))
    sustain_n = max(0, length - attack_n - decay_n - release_n)
    env = np.concatenate([
        np.linspace(0.0, 1.0, attack_n, endpoint=False, dtype=np.float32),
        np.linspace(1.0, sustain, decay_n, endpoint=False, dtype=np.float32),
        np.full(sustain_n, sustain, dtype=np.float32),
        np.linspace(sustain, 0.0, release_n, endpoint=True, dtype=np.float32),
    ])
    if env.size < length:
        env = np.pad(env, (0, length - env.size), mode='constant')
    return env[:length]


def _bass_tone(freq: float, t: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * freq * t)
        + 0.22 * np.sin(2 * np.pi * freq * 2.0 * t)
        + 0.08 * np.sin(2 * np.pi * freq * 3.0 * t)
    ) / 1.3


def _chord_tone(freq: float, t: np.ndarray) -> np.ndarray:
    tri = (2.0 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    sine = np.sin(2 * np.pi * freq * t)
    return (0.68 * tri + 0.32 * sine + 0.12 * np.sin(2 * np.pi * freq * 2.0 * t) + 0.04 * np.sin(2 * np.pi * freq * 3.0 * t)) / 1.16


def _add_signal(buffer: np.ndarray, signal: np.ndarray, start: int) -> None:
    if start >= buffer.size or signal.size <= 0:
        return
    end = min(buffer.size, start + signal.size)
    buffer[start:end] += signal[: end - start]


def synthesize_reference_wav_bytes(chords: list[str], bpm: float = 120.0, beats_per_chord: int = 4, target_duration_sec: Optional[float] = None) -> tuple[bytes, dict]:
    """Render a mono chord-reference WAV in memory and return debug metadata.

    Args:
        chords: Sequence of chord symbols to render. Empty inputs fall back to a
            default progression for preview generation.
        bpm: Target tempo in beats per minute. Values are clamped to a safe range.
        beats_per_chord: Number of beats assigned to each chord event.
        target_duration_sec: Optional target duration. Rendering is capped by
            ``MAX_RENDER_DURATION_SEC`` to avoid excessive allocations.

    Returns:
        tuple[bytes, dict]: PCM WAV bytes and metadata describing warnings,
        rendered events, effective duration, and loop count.
    """
    sample_rate = 44100
    safe_bpm = max(48.0, min(220.0, float(bpm or 120.0)))
    beat_sec = 60.0 / safe_bpm
    chord_sec = max(beat_sec * 2.0, beat_sec * max(1, int(beats_per_chord or 4)))
    requested = [str(x or '').strip() for x in (chords or []) if str(x or '').strip()]
    parsed_sequence: list[ParsedChord] = []
    warnings = []
    warning_debug = []
    for symbol in (requested or ['Cmaj7', 'Am7', 'Fmaj7', 'G']):
        parsed = parse_chord_symbol(symbol)
        if parsed is None:
            fallback_parsed = parse_chord_symbol('C')
            warnings.append({'symbol': symbol, 'reason': 'unparsed', 'fallback': 'C'})
            warning_debug.append(_warning_debug_entry(symbol, fallback_parsed, 'unparsed', 'C'))
            parsed = fallback_parsed
        elif parsed.warning:
            warnings.append({'symbol': symbol, 'reason': parsed.warning, 'fallback': parsed.descriptor})
            warning_debug.append(_warning_debug_entry(symbol, parsed, parsed.warning, parsed.descriptor))
        parsed_sequence.append(parsed)
    base_duration = len(parsed_sequence) * chord_sec
    requested_duration = max(base_duration, float(target_duration_sec or 0.0))
    capped_duration = min(requested_duration, MAX_RENDER_DURATION_SEC)
    if requested_duration > capped_duration:
        warnings.append({
            'symbol': '',
            'reason': 'duration_truncated',
            'fallback': str(MAX_RENDER_DURATION_SEC),
        })
        warning_debug.append({
            'symbol': '',
            'reason': 'duration_truncated',
            'fallback': str(MAX_RENDER_DURATION_SEC),
            'requested_duration_sec': round(requested_duration, 4),
            'capped_duration_sec': round(capped_duration, 4),
        })
    loop_count = max(1, math.ceil(capped_duration / max(base_duration, 0.001)))
    expanded = parsed_sequence * loop_count
    total_duration = max(base_duration, capped_duration)
    total_samples = max(1, int(sample_rate * total_duration))
    pcm = np.zeros(total_samples, dtype=np.float32)
    prev_pad = None
    prev_bass = None
    events = []
    for idx, chord in enumerate(expanded):
        start_sec = idx * chord_sec
        if start_sec >= total_duration:
            break
        dur_sec = min(chord_sec, total_duration - start_sec)
        start = int(start_sec * sample_rate)
        bass_midi, pad_midis = choose_voicing(chord, prev_pad, prev_bass)
        prev_pad = pad_midis
        prev_bass = bass_midi
        bass_len = max(1, int(sample_rate * max(0.18, dur_sec - 0.06)))
        bass_t = np.arange(bass_len, dtype=np.float32) / sample_rate
        bass_env = _envelope(bass_len, sample_rate, 0.012, 0.09, 0.82, 0.12)
        bass_sig = 0.23 * _bass_tone(midi_to_freq(bass_midi), bass_t) * bass_env
        _add_signal(pcm, bass_sig.astype(np.float32), start)
        chord_len = max(1, int(sample_rate * max(0.22, dur_sec - 0.04)))
        chord_t = np.arange(chord_len, dtype=np.float32) / sample_rate
        chord_env = _envelope(chord_len, sample_rate, 0.02, 0.12, 0.88, 0.16)
        for note_idx, midi in enumerate(pad_midis):
            stagger = int(sample_rate * 0.004 * note_idx)
            amp = 0.105 if note_idx == 0 else 0.12
            sig = amp * _chord_tone(midi_to_freq(midi), chord_t) * chord_env
            _add_signal(pcm, sig.astype(np.float32), start + stagger)
        if dur_sec >= beat_sec * 3.25:
            refresh_start = start + int(sample_rate * beat_sec * 2.5)
            refresh_len = max(1, int(sample_rate * min(beat_sec * 0.8, max(0.15, dur_sec - beat_sec * 2.5))))
            refresh_t = np.arange(refresh_len, dtype=np.float32) / sample_rate
            refresh_env = _envelope(refresh_len, sample_rate, 0.01, 0.06, 0.72, 0.10)
            for note_idx, midi in enumerate(pad_midis[:3]):
                sig = 0.045 * _chord_tone(midi_to_freq(midi), refresh_t) * refresh_env
                _add_signal(pcm, sig.astype(np.float32), refresh_start + int(sample_rate * 0.005 * note_idx))
        events.append({
            'index': idx,
            'symbol': chord.original,
            'normalized': chord.normalized,
            'start_sec': round(start_sec, 4),
            'dur_sec': round(dur_sec, 4),
            'bass_midi': bass_midi,
            'pad_midis': pad_midis,
        })
    fade = min(int(sample_rate * 0.02), max(1, total_samples // 12))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        pcm[:fade] *= ramp
        pcm[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 0:
        pcm *= min(0.92 / peak, 1.0)
    pcm16 = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue(), {
        'sample_rate': sample_rate,
        'bpm': safe_bpm,
        'beats_per_chord': int(max(1, beats_per_chord or 4)),
        'input_chords': requested,
        'warnings': warnings,
        'warning_count': len(warning_debug),
        'warning_debug': warning_debug,
        'rendered_events': events,
        'total_duration_sec': round(total_duration, 4),
        'loop_count': loop_count,
    }


def render_reference_wav_file(chords: list[str], output_path: str | Path, bpm: float = 120.0, beats_per_chord: int = 4, target_duration_sec: Optional[float] = None) -> dict:
    """Render a chord-reference WAV to disk and return the render metadata.

    Args:
        chords: Sequence of chord symbols to synthesize.
        output_path: Destination file path for the rendered WAV file.
        bpm: Target tempo in beats per minute.
        beats_per_chord: Number of beats assigned to each chord event.
        target_duration_sec: Optional target render duration, still subject to
            the module safety cap.

    Returns:
        dict: Metadata returned by :func:`synthesize_reference_wav_bytes` plus
        ``output_path`` and ``size_bytes`` for the written file.

    Raises:
        OSError: If the destination directory or file cannot be created.
    """
    wav_bytes, meta = synthesize_reference_wav_bytes(chords=chords, bpm=bpm, beats_per_chord=beats_per_chord, target_duration_sec=target_duration_sec)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)
    meta['output_path'] = str(out)
    meta['size_bytes'] = out.stat().st_size
    return meta
