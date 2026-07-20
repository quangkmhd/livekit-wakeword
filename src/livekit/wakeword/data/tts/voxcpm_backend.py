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
        use_gguf: bool = False,
        gguf_baselm_path: Path | None = None,
        gguf_acoustic_path: Path | None = None,
        voxcpm2_cli_path: Path | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._load_denoiser = load_denoiser
        self._prompts = voice_design_prompts
        self._cfg_values = cfg_values
        self._timesteps = inference_timesteps_list
        self._use_gguf = use_gguf
        self._gguf_baselm_path = gguf_baselm_path
        self._gguf_acoustic_path = gguf_acoustic_path
        self._voxcpm2_cli_path = voxcpm2_cli_path
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
            use_gguf=vt.use_gguf,
            gguf_baselm_path=config.voxcpm_gguf_baselm_path,
            gguf_acoustic_path=config.voxcpm_gguf_acoustic_path,
            voxcpm2_cli_path=config.voxcpm2_cli_path,
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
        if self._use_gguf:
            if (
                self._voxcpm2_cli_path is None
                or self._gguf_baselm_path is None
                or self._gguf_acoustic_path is None
            ):
                raise ValueError("GGUF paths are not configured in VoxCpmBackend")
            if not self._voxcpm2_cli_path.exists():
                raise FileNotFoundError(
                    f"voxcpm2-cli binary not found: {self._voxcpm2_cli_path}. "
                    "Run: livekit-wakeword setup --config <your.yaml>"
                )
            if not self._gguf_baselm_path.exists() or not self._gguf_acoustic_path.exists():
                raise FileNotFoundError(
                    f"GGUF model files not found. Expected: "
                    f"{self._gguf_baselm_path} and {self._gguf_acoustic_path}. "
                    "Run: livekit-wakeword setup --config <your.yaml>"
                )
            if not self._prompts or not self._cfg_values or not self._timesteps:
                raise ValueError(
                    "voxcpm_tts.voice_design_prompts, cfg_values, and "
                    "inference_timesteps_list must be non-empty"
                )
            return

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

        if self._use_gguf:
            import subprocess

            import librosa
            import soundfile as sf  # type: ignore[import-untyped]
            from tqdm import tqdm  # type: ignore[import-untyped]

            generated: list[Path] = []
            pbar = tqdm(
                range(start_index, n_samples),
                desc="VoxCPM GGUF clips",
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
                out_path = output_dir / f"clip_{sample_idx:06d}.wav"

                cmd = [
                    str(self._voxcpm2_cli_path),
                    "-t", text,
                    "-o", str(out_path),
                    "--cfg", f"{cfg_v:.2f}",
                    "--timesteps", str(steps),
                    str(self._gguf_baselm_path),
                    str(self._gguf_acoustic_path),
                ]
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )

                    if out_path.exists():
                        # Read generated wav (48kHz) and resample to 16kHz
                        audio, sr = sf.read(str(out_path))
                        audio = np.asarray(audio, dtype=np.float32).flatten()
                        if audio.size == 0:
                            logger.warning(
                                "VoxCPM GGUF returned empty audio at clip %d",
                                sample_idx,
                            )
                            try:
                                out_path.unlink()
                            except OSError:
                                pass
                            continue

                        if sr != TARGET_SAMPLE_RATE:
                            audio = librosa.resample(
                                audio,
                                orig_sr=sr,
                                target_sr=TARGET_SAMPLE_RATE,
                            )
                        peak = float(np.max(np.abs(audio))) or 1.0
                        audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)
                        sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
                        generated.append(out_path)
                    else:
                        logger.warning("voxcpm2-cli did not create output file at %s", out_path)
                except subprocess.TimeoutExpired as e:
                    logger.warning("VoxCPM GGUF generate timed out at clip %d: %s", sample_idx, e)
                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except OSError:
                            pass
                    continue
                except subprocess.CalledProcessError as e:
                    logger.warning("VoxCPM GGUF generate failed at clip %d: %s", sample_idx, e)
                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except OSError:
                            pass
                    continue

            logger.info("Generated %d GGUF clips in %s", len(generated), output_dir)
            return generated

        model = self._ensure_model()
        src_sr = int(model.tts_model.sample_rate)

        import librosa
        import soundfile as sf
        from tqdm import tqdm

        generated = []
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
            peak = float(np.max(np.abs(audio))) or 1.0
            audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)

            out_path = output_dir / f"clip_{sample_idx:06d}.wav"
            sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
            generated.append(out_path)

        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return generated
