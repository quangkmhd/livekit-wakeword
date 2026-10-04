"""VoxCPM2 TTS backend: voice-design diversification (persona × cfg × diffusion steps)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...config import WakeWordConfig

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    top_db: float = 25.0,
    frame_length: int = 512,
    hop_length: int = 160,
    max_cluster_gap_s: float = 0.30,
    max_internal_pause_s: float = 0.18,
    target_internal_pause_s: float = 0.12,
    max_duration_s: float = 2.0,
) -> np.ndarray:
    """Trim silence, drop trailing TTS hallucinations, and cap long internal pauses.

    1. Splits audio into non-silent intervals via ``librosa.effects.split`` and filters
       out micro-clicks (< 40 ms or RMS < 15% of the loudest segment).
    2. Groups intervals separated by ``<= max_cluster_gap_s`` into speech clusters.
       Once the main phrase at the start has been spoken (>= 450 ms of voiced speech),
       any subsequent cluster separated by a silence gap (> 300 ms) is unconditionally
       dropped as an end-of-sequence hallucination (even if the hallucination is 4-8s long).
    3. Preserves natural inter-word pauses (``<= max_internal_pause_s``) intact, while
       compressing abnormally long internal pauses down to ``target_internal_pause_s``
       and capping total output duration to ``max_duration_s`` from the front so the
       initial wake word is never pushed out of the 2-second training window.
    """
    import librosa

    raw_intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(raw_intervals) == 0:
        return audio

    min_seg_samples = int(sample_rate * 0.04)
    seg_rms = [
        float(np.sqrt(np.mean(audio[s:e] ** 2))) if e > s else 0.0
        for s, e in raw_intervals
    ]
    max_rms = max(seg_rms) if seg_rms else 0.0
    valid_intervals: list[tuple[int, int]] = [
        (int(s), int(e))
        for (s, e), r in zip(raw_intervals, seg_rms)
        if (e - s) >= min_seg_samples and r >= 0.15 * max_rms
    ]
    if not valid_intervals:
        valid_intervals = [(int(raw_intervals[0][0]), int(raw_intervals[-1][1]))]

    max_gap_samples = int(sample_rate * max_cluster_gap_s)
    clusters: list[list[tuple[int, int]]] = [[valid_intervals[0]]]
    for s, e in valid_intervals[1:]:
        prev_e = clusters[-1][-1][1]
        if (s - prev_e) <= max_gap_samples:
            clusters[-1].append((s, e))
        else:
            clusters.append([(s, e)])

    kept_intervals: list[tuple[int, int]] = list(clusters[0])
    preceding_voiced = sum(e - s for s, e in kept_intervals)
    min_complete_phrase_samples = int(sample_rate * 0.45)

    for cluster in clusters[1:]:
        if preceding_voiced >= min_complete_phrase_samples:
            # Main wake word phrase at the beginning is already complete; drop all
            # subsequent clusters regardless of how long the trailing hallucination is.
            break
        kept_intervals.extend(cluster)
        preceding_voiced += sum(e - s for s, e in cluster)

    max_pause_samples = int(sample_rate * max_internal_pause_s)
    target_pause_samples = int(sample_rate * target_internal_pause_s)
    max_total_samples = int(sample_rate * max_duration_s)

    pieces: list[np.ndarray] = [audio[kept_intervals[0][0] : kept_intervals[0][1]]]
    accumulated_len = len(pieces[0])
    accumulated_voiced = len(pieces[0])

    for idx in range(1, len(kept_intervals)):
        prev_e = kept_intervals[idx - 1][1]
        cur_s, cur_e = kept_intervals[idx]
        gap = cur_s - prev_e
        pause_len = gap if gap <= max_pause_samples else target_pause_samples
        seg_len = cur_e - cur_s

        # If main phrase is already spoken and adding another segment after a pause
        # would overflow the 16-frame (~1.5s) budget, stop before the extra babble.
        if (
            accumulated_voiced >= min_complete_phrase_samples
            and gap >= int(sample_rate * 0.15)
            and (accumulated_len + pause_len + seg_len) > int(sample_rate * 1.5)
        ):
            break

        if gap <= max_pause_samples:
            pieces.append(audio[prev_e:cur_e])
        else:
            pieces.append(audio[prev_e : prev_e + target_pause_samples])
            pieces.append(audio[cur_s:cur_e])
        accumulated_len += pause_len + seg_len
        accumulated_voiced += seg_len

    trimmed = np.concatenate(pieces)
    if len(trimmed) > max_total_samples:
        trimmed = trimmed[:max_total_samples]

    min_speech_samples = int(sample_rate * 0.15)
    if len(trimmed) < min_speech_samples:
        logger.debug(
            "Silence trim stripped too aggressively (%d samples left), keeping original",
            len(trimmed),
        )
        return audio
    return trimmed


def diversification_triple_at_index(
    prompts: list[str],
    cfg_values: list[float],
    timesteps: list[int],
    index: int,
) -> tuple[str, float, int]:
    """Return the (prompt, cfg, steps) triple for global clip *index* (resume-safe).

    Ordering matches ``itertools.product(prompts, cfg_values, timesteps)``:
    innermost dimension is *timesteps*, then *cfg_values*, then *prompts*.
    """
    np_ = len(prompts)
    nc = len(cfg_values)
    nt = len(timesteps)
    n = np_ * nc * nt
    if n == 0:
        raise ValueError("voxcpm diversification lists must be non-empty")
    flat = index % n
    ti = flat % nt
    flat //= nt
    ci = flat % nc
    pi = flat // nc
    return prompts[pi], cfg_values[ci], timesteps[ti]


class VoxCpmBackend:
    """VoxCPM2 with strong default diversification; loads weights from local snapshot only."""

    def __init__(
        self,
        *,
        model_dir: Path,
        load_denoiser: bool,
        voice_design_prompts: list[str],
        cfg_values: list[float],
        inference_timesteps_list: list[int],
    ) -> None:
        self._model_dir = model_dir
        self._load_denoiser = load_denoiser
        self._prompts = voice_design_prompts
        self._cfg_values = cfg_values
        self._timesteps = inference_timesteps_list
        self._model: Any = None

    @classmethod
    def from_config(cls, config: WakeWordConfig) -> VoxCpmBackend:
        vt = config.voxcpm_tts
        return cls(
            model_dir=config.voxcpm_local_model_path,
            load_denoiser=vt.load_denoiser,
            voice_design_prompts=list(vt.voice_design_prompts),
            cfg_values=list(vt.cfg_values),
            inference_timesteps_list=list(vt.inference_timesteps_list),
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if importlib.util.find_spec("voxcpm") is None:
            raise ImportError(
                "VoxCPM is not installed. Install with: uv sync --extra train --extra voxcpm"
            )
        voxcpm_mod = importlib.import_module("voxcpm")
        VoxCPM = getattr(voxcpm_mod, "VoxCPM")
        logger.info("Loading VoxCPM from %s", self._model_dir)
        self._model = VoxCPM.from_pretrained(
            str(self._model_dir),
            load_denoiser=self._load_denoiser,
        )
        return self._model

    def validate_artifacts(self) -> None:
        if not self._model_dir.is_dir():
            raise FileNotFoundError(
                f"VoxCPM model directory not found: {self._model_dir}. "
                "Run: livekit-wakeword setup --config <your.yaml>"
            )
        if not any(self._model_dir.iterdir()):
            raise FileNotFoundError(
                f"VoxCPM model directory is empty: {self._model_dir}. "
                "Run: livekit-wakeword setup --config <your.yaml>"
            )
        if not self._prompts or not self._cfg_values or not self._timesteps:
            raise ValueError(
                "voxcpm_tts.voice_design_prompts, cfg_values, and "
                "inference_timesteps_list must be non-empty"
            )
        if importlib.util.find_spec("voxcpm") is None:
            raise ImportError(
                "VoxCPM is not installed. Install with: uv sync --extra train --extra voxcpm"
            )

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        del batch_size  # sequential generation only
        if not phrases:
            raise ValueError("phrases must be non-empty")
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._ensure_model()
        src_sr = int(model.tts_model.sample_rate)

        import librosa
        import soundfile as sf  # type: ignore[import-untyped]
        from tqdm import tqdm  # type: ignore[import-untyped]

        generated: list[Path] = []
        pbar = tqdm(
            range(start_index, n_samples),
            desc="VoxCPM clips",
            unit="clip",
            initial=start_index,
            total=n_samples,
        )
        for sample_idx in pbar:
            phrase = phrases[sample_idx % len(phrases)]
            prompt, cfg_v, steps = diversification_triple_at_index(
                self._prompts,
                self._cfg_values,
                self._timesteps,
                sample_idx,
            )
            text = f"({prompt}){phrase}"
            try:
                wav = model.generate(
                    text=text,
                    cfg_value=cfg_v,
                    inference_timesteps=steps,
                )
            except Exception as e:
                logger.warning("VoxCPM generate failed at clip %d: %s", sample_idx, e)
                continue

            audio = np.asarray(wav, dtype=np.float32).flatten()
            if audio.size == 0:
                logger.warning("VoxCPM returned empty audio at clip %d", sample_idx)
                continue

            if src_sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=src_sr,
                    target_sr=TARGET_SAMPLE_RATE,
                )
            audio = trim_silence(audio, sample_rate=TARGET_SAMPLE_RATE)
            peak = float(np.max(np.abs(audio))) or 1.0
            audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)

            out_path = output_dir / f"clip_{sample_idx:06d}.wav"
            sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
            generated.append(out_path)

        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return generated
