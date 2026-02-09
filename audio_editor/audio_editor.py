import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

try:
    import librosa
except Exception:
    librosa = None


def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    return wav.float(), sr


def _match_channels(wav: torch.Tensor, target_channels: int) -> torch.Tensor:
    channels = wav.shape[0]
    if channels == target_channels:
        return wav
    if target_channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if channels == 1 and target_channels > 1:
        return wav.repeat(target_channels, 1)
    raise ValueError(f"Channel mismatch: {channels} -> {target_channels} not supported")


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def _concat_audios(paths: List[str]) -> Tuple[torch.Tensor, int]:
    if not paths:
        raise ValueError("No input audio files provided.")

    base_wav, base_sr = _load_audio(paths[0])
    base_channels = base_wav.shape[0]
    segments = [base_wav]

    for path in paths[1:]:
        wav, sr = _load_audio(path)
        wav = _match_channels(wav, base_channels)
        wav = _resample(wav, sr, base_sr)
        segments.append(wav)

    merged = torch.cat(segments, dim=1)
    return merged, base_sr


def _apply_speed(
    wav: torch.Tensor,
    sr: int,
    speed: float,
    preserve_pitch: bool,
) -> Tuple[torch.Tensor, int]:
    if speed <= 0:
        raise ValueError("Speed must be greater than 0.")
    if speed == 1.0:
        return wav, sr

    if preserve_pitch:
        if librosa is None:
            raise RuntimeError("librosa is required for --preserve-pitch but is not installed.")
        wav_np = wav.cpu().numpy()
        stretched_channels = []
        for ch in wav_np:
            stretched = librosa.effects.time_stretch(ch, rate=speed)
            stretched_channels.append(stretched)
        max_len = max(len(ch) for ch in stretched_channels)
        padded = [np.pad(ch, (0, max_len - len(ch)), mode="constant") for ch in stretched_channels]
        return torch.from_numpy(np.stack(padded, axis=0)).float(), sr

    new_sr = int(round(sr / speed))
    if new_sr <= 0:
        raise ValueError("Speed is too large for resampling.")
    sped = torchaudio.functional.resample(wav, sr, new_sr)
    return sped, sr


def _parse_silence_entries(entries: List[str]) -> List[Tuple[float, float]]:
    result = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid silence format: {entry!r}. Use time:duration (seconds).")
        time_s, dur_s = entry.split(":", 1)
        time_val = float(time_s)
        dur_val = float(dur_s)
        if time_val < 0 or dur_val < 0:
            raise ValueError(f"Silence time/duration must be >= 0: {entry!r}")
        result.append((time_val, dur_val))
    return result


def _insert_silences(
    wav: torch.Tensor,
    sr: int,
    silences: List[Tuple[float, float]],
) -> torch.Tensor:
    if not silences:
        return wav

    silences_sorted = sorted(silences, key=lambda x: x[0])
    total_inserted = 0.0
    for time_s, dur_s in silences_sorted:
        if dur_s == 0:
            continue
        insert_at = time_s + total_inserted
        idx = int(round(insert_at * sr))
        idx = max(0, min(idx, wav.shape[1]))
        silence_samples = int(round(dur_s * sr))
        silence = torch.zeros(wav.shape[0], silence_samples, dtype=wav.dtype)
        wav = torch.cat([wav[:, :idx], silence, wav[:, idx:]], dim=1)
        total_inserted += dur_s
    return wav


def _save_audio(path: str, wav: torch.Tensor, sr: int) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(path, wav, sr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple audio editor: speed, concat, and insert silence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input audio files in order.")
    parser.add_argument("--output", required=True, help="Output audio file path.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor.")
    parser.add_argument(
        "--preserve-pitch",
        action="store_true",
        default=False,
        help="Preserve pitch using time-stretch (requires librosa).",
    )
    parser.add_argument(
        "--insert-silence",
        action="append",
        default=[],
        help="Insert silence: time:duration (seconds). Can be used multiple times.",
    )

    args = parser.parse_args()

    wav, sr = _concat_audios(args.input)
    wav, sr = _apply_speed(wav, sr, args.speed, args.preserve_pitch)
    silences = _parse_silence_entries(args.insert_silence)
    wav = _insert_silences(wav, sr, silences)
    _save_audio(args.output, wav, sr)

    duration = wav.shape[1] / sr
    print(f"Saved: {args.output} ({duration:.2f}s, {sr}Hz)")


if __name__ == "__main__":
    main()
