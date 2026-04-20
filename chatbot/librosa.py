from __future__ import annotations

from typing import Any

import numpy as np


_KEY_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Match user's original analysis script defaults
_SR = 22050
_N_FFT = 2048
_HOP_LENGTH = 512


def _init_key_ohe() -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _KEY_NAMES:
        out[f"key_{k}:maj"] = 0.0
        out[f"key_{k}:min"] = 0.0
    return out


def _key_ohe_from_musical_key(musical_key: str) -> dict[str, float]:
    out = _init_key_ohe()
    s = str(musical_key or "").strip()
    if not s or s.lower() == "unknown" or ":" not in s:
        return out
    try:
        k, mode = s.split(":", 1)
        k = k.strip()
        mode = mode.strip().lower()
        if mode in {"maj", "min"} and k in _KEY_NAMES:
            out[f"key_{k}:{mode}"] = 1.0
    except Exception:
        return out
    return out


def extract_audio_features_full(audio_path: str, *, sr: int = 22050) -> dict[str, Any]:
    """Extract full-song audio features using librosa.

    This function intentionally mirrors the user's original offline feature script
    (see `librosa_colab.py`) for consistent feature values.

    Best-effort: never raises; returns `audio_error` on failure.
    """

    record: dict[str, Any] = {}

    # Always use the same constants as the user's script unless explicitly overridden.
    sr_use_target = int(sr) if sr else int(_SR)

    try:
        import librosa

        y, sr_loaded = librosa.load(audio_path, sr=int(sr_use_target), mono=True)
        sr_use = int(sr_loaded)

        record["duration_sec"] = round(float(librosa.get_duration(y=y, sr=sr_use)), 2)
        record["rms_energy"] = round(
            float(np.mean(librosa.feature.rms(y=y, frame_length=_N_FFT, hop_length=_HOP_LENGTH))),
            6,
        )
        record["zero_crossing_rate"] = round(
            float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=_N_FFT, hop_length=_HOP_LENGTH))),
            6,
        )

        cent = librosa.feature.spectral_centroid(y=y, sr=sr_use, n_fft=_N_FFT, hop_length=_HOP_LENGTH)[0]
        record["spectral_centroid_mean"] = round(float(np.mean(cent)), 2)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_use, n_fft=_N_FFT, hop_length=_HOP_LENGTH)[0]
        record["spectral_rolloff"] = round(float(np.mean(rolloff)), 2)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr_use, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        for i in range(int(contrast.shape[0])):
            record[f"spectral_contrast_band{i + 1}_mean"] = round(float(np.mean(contrast[i])), 2)

        record["spectral_flatness_mean"] = round(
            float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=_N_FFT, hop_length=_HOP_LENGTH))),
            6,
        )

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr_use, hop_length=_HOP_LENGTH)
        record["tempo_bpm"] = round(float(tempo), 2)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr_use, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        record["beat_strength_mean"] = round(float(np.mean(onset_env)), 4)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_use, hop_length=_HOP_LENGTH)
        record["onset_rate"] = round(float(len(onset_frames) / max(record.get("duration_sec", 0.0) or 0.0, 1)), 2)

        if beats is not None and len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr_use)
            beat_intervals = np.diff(beat_times)
            m = float(np.mean(beat_intervals)) if beat_intervals.size else 0.0
            s = float(np.std(beat_intervals)) if beat_intervals.size else 0.0
            record["tempo_stability"] = round(float(1 - (s / m)), 4) if m > 0 else 0.0
        else:
            record["tempo_stability"] = 0.0

        mfccs = librosa.feature.mfcc(y=y, sr=sr_use, n_mfcc=13, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        for i in range(13):
            record[f"mfcc{i + 1}_mean"] = round(float(np.mean(mfccs[i])), 4)
            record[f"mfcc{i + 1}_std"] = round(float(np.std(mfccs[i])), 4)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr_use, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        for i in range(12):
            record[f"chroma{i + 1}_mean"] = round(float(np.mean(chroma[i])), 4)

        y_harmonic = librosa.effects.harmonic(y)
        y_perc = librosa.effects.percussive(y)
        record["harmonic_percussive_ratio"] = round(
            float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_perc)) + 1e-10)),
            4,
        )

        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr_use)
        for i in range(6):
            record[f"tonnetz{i + 1}_mean"] = round(float(np.mean(tonnetz[i])), 4)

        chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr_use)
        chroma_mean = np.mean(chroma_cens, axis=1)
        major_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-10)
        major_p = major_p / np.linalg.norm(major_p)
        minor_p = minor_p / np.linalg.norm(minor_p)
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_key, max_corr = "Unknown", -1.0
        for i in range(12):
            c_maj = float(np.dot(chroma_mean, np.roll(major_p, i)))
            c_min = float(np.dot(chroma_mean, np.roll(minor_p, i)))
            if c_maj > max_corr:
                max_corr, best_key = c_maj, f"{pitch_names[i]}:maj"
            if c_min > max_corr:
                max_corr, best_key = c_min, f"{pitch_names[i]}:min"
        record["musical_key"] = str(best_key)

        # Optional compatibility: also expose key one-hot columns.
        record.update(_key_ohe_from_musical_key(record.get("musical_key", "")))

    except Exception as ex:
        record["audio_error"] = f"{type(ex).__name__}: {ex}"
        # Best-effort: populate expected keys with safe defaults
        record.setdefault("duration_sec", 0.0)
        record.setdefault("tempo_bpm", 0.0)
        record.setdefault("musical_key", "Unknown")
        record.setdefault("rms_energy", 0.0)
        record.setdefault("spectral_centroid_mean", 0.0)
        for i in range(1, 14):
            record.setdefault(f"mfcc{i}_mean", 0.0)
            record.setdefault(f"mfcc{i}_std", 0.0)
        record.setdefault("zero_crossing_rate", 0.0)
        record.setdefault("spectral_rolloff", 0.0)
        for i in range(1, 8):
            record.setdefault(f"spectral_contrast_band{i}_mean", 0.0)
        record.setdefault("spectral_flatness_mean", 0.0)
        record.setdefault("beat_strength_mean", 0.0)
        record.setdefault("onset_rate", 0.0)
        record.setdefault("tempo_stability", 0.0)
        for i in range(1, 13):
            record.setdefault(f"chroma{i}_mean", 0.0)
        for i in range(1, 7):
            record.setdefault(f"tonnetz{i}_mean", 0.0)
        record.setdefault("harmonic_percussive_ratio", 0.0)
        record.update(_key_ohe_from_musical_key(record.get("musical_key", "Unknown")))

    return record
