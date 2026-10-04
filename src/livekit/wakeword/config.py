"""Pydantic configuration models with YAML loading."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from .defaults import piper as piper_defaults
from .defaults import voxcpm as voxcpm_defaults

_logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    dnn = "dnn"
    rnn = "rnn"
    conv_attention = "conv_attention"


class ModelSize(StrEnum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"


class TtsBackend(StrEnum):
    """Synthetic speech engine for the generate stage."""

    piper_vits = "piper_vits"
    voxcpm = "voxcpm"


class ExportFormat(StrEnum):
    """Artifact format produced by the export stage."""

    onnx = "onnx"
    tflite = "tflite"


# Preset mapping: size -> (layer_dim, n_blocks)
MODEL_SIZE_PRESETS: dict[ModelSize, tuple[int, int]] = {
    ModelSize.tiny: (16, 1),
    ModelSize.small: (32, 1),
    ModelSize.medium: (128, 2),
    ModelSize.large: (256, 3),
}


class AugmentationConfig(BaseModel):
    clip_duration: float = 2.0
    batch_size: int = 16
    rounds: int = 1
    background_paths: list[str] = Field(default_factory=lambda: ["./data/backgrounds"])
    rir_paths: list[str] = Field(default_factory=lambda: ["./data/rirs"])


class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.conv_attention
    model_size: ModelSize = ModelSize.small

    @property
    def layer_dim(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][0]

    @property
    def n_blocks(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][1]


class PiperTtsConfig(BaseModel):
    """Piper VITS artifact layout under ``data_dir`` (when ``tts_backend`` is ``piper_vits``)."""

    checkpoint_relpath: str = Field(
        default=piper_defaults.CHECKPOINT_RELPATH,
        description="Path to the VITS state_dict .pt file, relative to data_dir",
    )


class VoxCpmTtsConfig(BaseModel):
    """VoxCPM2 voice-design TTS (when ``tts_backend`` is ``voxcpm``).

    Weights live under ``data_dir`` after ``setup --config`` (HF snapshot) or
    ``local_model_path``. Diversification defaults are intentionally large
    (persona × cfg × diffusion steps).
    """

    model_id: str = Field(
        default=voxcpm_defaults.MODEL_ID,
        description="Hugging Face repo id used by setup for snapshot_download",
    )
    model_cache_relpath: str = Field(
        default=voxcpm_defaults.MODEL_CACHE_RELPATH,
        description="Directory under data_dir where setup stores the model snapshot",
    )
    local_model_path: str | None = Field(
        default=None,
        description="If set, load weights from this path (relative to data_dir or absolute); "
        "setup skips HF download if directory exists and is non-empty",
    )
    load_denoiser: bool = False
    voice_design_prompts: list[str] = Field(default_factory=voxcpm_defaults.voice_design_prompts)
    voice_design_prompts_file: str | None = Field(
        default=None,
        description="Optional path to a .yaml or .txt file containing voice_design_prompts. "
        "When set, replaces default prompts (any inline voice_design_prompts are appended).",
    )
    cfg_values: list[float] = Field(default_factory=lambda: list(voxcpm_defaults.CFG_VALUES))
    inference_timesteps_list: list[int] = Field(
        default_factory=lambda: list(voxcpm_defaults.INFERENCE_TIMESTEPS),
    )

    @model_validator(mode="after")
    def _load_prompts_from_file(self) -> Self:
        if not self.voice_design_prompts_file:
            return self
        path = Path(self.voice_design_prompts_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"voice_design_prompts_file not found: {path}"
            )
        file_prompts = _load_string_list_file(path, yaml_key="voice_design_prompts")
        default_prompts = voxcpm_defaults.voice_design_prompts()
        if self.voice_design_prompts == default_prompts:
            self.voice_design_prompts = file_prompts
        else:
            self.voice_design_prompts = file_prompts + self.voice_design_prompts
        return self


def _load_string_list_file(path: Path, yaml_key: str | None = None) -> list[str]:
    """Load a list of non-empty strings from a .yaml/.yml or plain-text .txt file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict) and yaml_key and yaml_key in loaded:
            items = loaded[yaml_key]
        elif isinstance(loaded, list):
            items = loaded
        else:
            raise ValueError(
                f"Expected a YAML list or mapping with key {yaml_key!r} in {path}"
            )
        return [str(x).strip() for x in items if str(x).strip()]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class WakeWordConfig(BaseModel):
    """Top-level config for a wake word model."""

    model_name: str
    target_phrases: list[str]

    # Data generation
    n_samples: int = 10000
    n_samples_val: int = 2000
    n_background_samples: int = 200
    n_background_samples_val: int = 40
    tts_batch_size: int = 50
    tts_backend: TtsBackend = TtsBackend.piper_vits
    piper_tts: PiperTtsConfig = Field(default_factory=PiperTtsConfig)
    voxcpm_tts: VoxCpmTtsConfig = Field(default_factory=VoxCpmTtsConfig)
    auto_adversarial_negatives: bool = Field(
        default=True,
        description="Whether to auto-generate English CMUDict phoneme adversarial phrases. "
        "Set to false for non-English wake words (e.g. Vietnamese) to avoid English false matches.",
    )
    custom_negative_phrases: list[str] = Field(default_factory=list)
    custom_negative_phrases_file: str | None = Field(
        default=None,
        description="Optional path to a .txt or .yaml file of negative phrases to append to "
        "custom_negative_phrases.",
    )

    # TTS parameters (Piper VITS + SLERP speaker blending)
    noise_scales: list[float] = Field(default_factory=lambda: [0.98])
    noise_scale_ws: list[float] = Field(default_factory=lambda: [0.98])
    length_scales: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25])
    slerp_weights: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65, 0.8])
    max_speakers: int | None = None

    # Paths
    data_dir: Annotated[str, Field(description="Root data directory")] = "./data"
    dataset_dir: str | None = Field(
        default=None,
        description="Directory for generated WAV splits and extracted .npy features. "
        "When set, allows multiple model versions (v1, v2, ...) to reuse the same dataset "
        "without overwriting model outputs or re-running TTS. Defaults to model_output_dir.",
    )
    output_dir: str = "./output"

    # Export
    output_format: ExportFormat = ExportFormat.onnx

    # Augmentation
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # Model
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training
    steps: int = 50000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    max_negative_weight: float = 1500.0
    target_fp_per_hour: float = 0.2
    batch_n_per_class: dict[str, int] = Field(
        default_factory=lambda: {
            "positive": 50,
            "adversarial_negative": 50,
            "ACAV100M_sample": 1024,
            "background_noise": 50,
        }
    )

    @model_validator(mode="after")
    def _warn_unknown_batch_keys_and_load_files(self) -> Self:
        known_keys = {"positive", "adversarial_negative", "ACAV100M_sample", "background_noise"}
        unknown = set(self.batch_n_per_class) - known_keys
        if unknown:
            _logger.warning(
                f"Unrecognized keys in batch_n_per_class: {unknown}. "
                f"Known keys: {sorted(known_keys)}"
            )
        if self.custom_negative_phrases_file:
            neg_path = Path(self.custom_negative_phrases_file)
            if not neg_path.is_file():
                raise FileNotFoundError(
                    f"custom_negative_phrases_file not found: {neg_path}"
                )
            extra_phrases = _load_string_list_file(
                neg_path, yaml_key="custom_negative_phrases"
            )
            seen = set(self.custom_negative_phrases)
            for phrase in extra_phrases:
                if phrase not in seen:
                    self.custom_negative_phrases.append(phrase)
                    seen.add(phrase)
        return self

    @property
    def model_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name

    @property
    def dataset_output_dir(self) -> Path:
        """Directory containing WAV splits and extracted .npy feature files."""
        if self.dataset_dir:
            return Path(self.dataset_dir)
        return self.model_output_dir

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def piper_checkpoint_path(self) -> Path:
        """Absolute path to the Piper VITS .pt checkpoint (JSON sits alongside)."""
        return (self.data_path / Path(self.piper_tts.checkpoint_relpath)).resolve()

    @property
    def voxcpm_local_model_path(self) -> Path:
        """Directory containing VoxCPM weights (snapshot or manual copy)."""
        raw = self.voxcpm_tts.local_model_path
        if raw:
            p = Path(raw)
            return p.resolve() if p.is_absolute() else (self.data_path / p).resolve()
        return (self.data_path / Path(self.voxcpm_tts.model_cache_relpath)).resolve()


def load_config(path: str | Path) -> WakeWordConfig:
    """Load a WakeWordConfig from a YAML file."""
    cfg_path = Path(path).resolve()
    cfg_dir = cfg_path.parent
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Resolve relative file references against cwd first, then config parent directory,
    # then repo root (cfg_dir.parent when config is in configs/).
    def _resolve_aux_file(raw_path: str | None) -> str | None:
        if not raw_path:
            return raw_path
        p = Path(raw_path)
        if p.is_file():
            return str(p)
        for base in (cfg_dir, cfg_dir.parent):
            cand = base / p
            if cand.is_file():
                return str(cand)
        return raw_path

    if isinstance(data, dict):
        if "custom_negative_phrases_file" in data:
            data["custom_negative_phrases_file"] = _resolve_aux_file(
                data["custom_negative_phrases_file"]
            )
        vox = data.get("voxcpm_tts")
        if isinstance(vox, dict) and "voice_design_prompts_file" in vox:
            vox["voice_design_prompts_file"] = _resolve_aux_file(
                vox["voice_design_prompts_file"]
            )

    return WakeWordConfig(**data)


