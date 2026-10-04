"""VoxCPM diversification index must match ``itertools.product`` (resume-safe)."""

from __future__ import annotations

import itertools as it
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from livekit.wakeword.data.tts.voxcpm_backend import (
    TARGET_SAMPLE_RATE,
    VoxCpmBackend,
    diversification_triple_at_index,
    trim_silence,
)


def test_diversification_triple_matches_product_order() -> None:
    prompts = ["P0", "P1", "P2"]
    cfg_values = [1.5, 2.0]
    timesteps = [8, 10, 12]
    flat = list(it.product(prompts, cfg_values, timesteps))
    n = len(prompts) * len(cfg_values) * len(timesteps)
    for i in range(n * 3 + 7):
        assert diversification_triple_at_index(prompts, cfg_values, timesteps, i) == flat[i % n]


def test_voxcpm_synthesize_clips_trims_silence(tmp_path: Path) -> None:
    backend = VoxCpmBackend(
        model_dir=tmp_path,
        load_denoiser=False,
        voice_design_prompts=["Persona"],
        cfg_values=[2.0],
        inference_timesteps_list=[10],
    )
    fake_raw_audio = np.linspace(-0.5, 0.5, TARGET_SAMPLE_RATE, dtype=np.float32)
    fake_model = SimpleNamespace(
        tts_model=SimpleNamespace(sample_rate=TARGET_SAMPLE_RATE),
        generate=lambda text, cfg_value, inference_timesteps: fake_raw_audio,
    )
    backend._model = fake_model

    expected_trimmed = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    with patch(
        "livekit.wakeword.data.tts.voxcpm_backend.trim_silence",
        return_value=expected_trimmed,
    ) as mock_trim_silence:
        out_dir = tmp_path / "clips"
        paths = backend.synthesize_clips(["hey livekit"], out_dir, n_samples=1)

    assert len(paths) == 1
    mock_trim_silence.assert_called_once()
    _, call_kwargs = mock_trim_silence.call_args
    assert call_kwargs.get("sample_rate") == TARGET_SAMPLE_RATE

    saved_audio, sr = sf.read(str(paths[0]), dtype="int16")
    assert sr == TARGET_SAMPLE_RATE
    assert len(saved_audio) == len(expected_trimmed)


def test_voxcpm_synthesize_clips_removes_trailing_silence_end_to_end(tmp_path: Path) -> None:
    backend = VoxCpmBackend(
        model_dir=tmp_path,
        load_denoiser=False,
        voice_design_prompts=["Persona"],
        cfg_values=[2.0],
        inference_timesteps_list=[10],
    )
    # 0.5s speech-like signal + 0.5s low-level AudioVAE-like tail noise (-50 dBFS)
    t = np.linspace(0, 0.5, TARGET_SAMPLE_RATE // 2, endpoint=False, dtype=np.float32)
    voiced = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    rng = np.random.default_rng(0)
    tail_noise = rng.normal(0.0, 1e-3, size=TARGET_SAMPLE_RATE // 2).astype(np.float32)
    raw_audio = np.concatenate([voiced, tail_noise])

    backend._model = SimpleNamespace(
        tts_model=SimpleNamespace(sample_rate=TARGET_SAMPLE_RATE),
        generate=lambda text, cfg_value, inference_timesteps: raw_audio,
    )

    out_dir = tmp_path / "clips_e2e"
    paths = backend.synthesize_clips(["hey livekit"], out_dir, n_samples=1)

    assert len(paths) == 1
    saved_audio, sr = sf.read(str(paths[0]), dtype="int16")
    assert sr == TARGET_SAMPLE_RATE
    assert len(saved_audio) <= TARGET_SAMPLE_RATE // 2 + 512


def test_trim_silence_drops_trailing_hallucination_and_compresses_long_pause() -> None:
    sr = TARGET_SAMPLE_RATE
    t_an = np.linspace(0, 0.27, int(0.27 * sr), endpoint=False, dtype=np.float32)
    an = 0.6 * np.sin(2 * np.pi * 220 * t_an)
    t_vy = np.linspace(0, 0.38, int(0.38 * sr), endpoint=False, dtype=np.float32)
    vy_oi = 0.6 * np.sin(2 * np.pi * 260 * t_vy)
    t_tail = np.linspace(0, 0.18, int(0.18 * sr), endpoint=False, dtype=np.float32)
    hallucination = 0.6 * np.sin(2 * np.pi * 300 * t_tail)

    # Case 1: Natural 110ms pause between "an" & "vy ơi" + 660ms gap + 180ms trailing hallucination
    natural_gap = np.zeros(int(0.11 * sr), dtype=np.float32)
    long_gap = np.zeros(int(0.66 * sr), dtype=np.float32)
    audio_with_hallucination = np.concatenate([an, natural_gap, vy_oi, long_gap, hallucination])

    cleaned = trim_silence(audio_with_hallucination, sample_rate=sr)
    # Must drop the 660ms gap + 180ms hallucination (~0.76s remaining)
    assert abs(len(cleaned) / sr - 0.76) < 0.04

    # Case 2: Abnormally long 450ms pause between "an" & "vy ơi" -> compressed to ~120ms
    pause_450ms = np.zeros(int(0.45 * sr), dtype=np.float32)
    audio_long_pause = np.concatenate([an, pause_450ms, vy_oi])
    compressed = trim_silence(audio_long_pause, sample_rate=sr)
    assert abs(len(compressed) / sr - 0.77) < 0.04

