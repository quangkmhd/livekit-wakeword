"""Evaluate wake word model on VietSuperSpeech dataset using GPU and original WakeWordModel architecture.

 VietSuperSpeech is a negative dataset (no wake words).
 Fast GPU vectorized evaluation:
 - MelSpectrogramFrontend ONNX on full audio (1 GPU call per file)
 - SpeechEmbedding ONNX sliding window (1 batched GPU call per file)
 - Classifier ONNX on all (N_windows, 16, 96) sequences (1 batched GPU call per file)
 - Saves detailed False Positive logs (file, start_sec, end_sec, score) for listening back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livekit.wakeword.resources import get_mel_model_path, get_embedding_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import argparse

THRESHOLDS = [0.5, 0.57, 0.65, 0.70]
MIN_LOG_THRESHOLD = 0.5  # Record all false positives with score >= 0.50
SAMPLE_RATE = 16000
EMBEDDING_WINDOW = 76
EMBEDDING_STRIDE = 8  # 8 mel frames = 80ms stride (1280 samples)
N_TIMESTEPS = 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wake word model on VietSuperSpeech")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/quangnhvn34/dev/oss/livekit-wakeword/output/chao_an_vi",
        help="Path to the model directory containing chao_an_vi.onnx"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/quangnhvn34/dev/oss/livekit-wakeword/VietSuperSpeech",
        help="Path to the VietSuperSpeech dataset"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    output_json = model_dir / "chao_an_vi_viet_speech_eval_gpu.json"
    fp_log_json = model_dir / "chao_an_vi_viet_speech_false_positives.json"

    logger.info("Setting up GPU execution providers (CUDAExecutionProvider)...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    classifier_path = model_dir / "chao_an_vi.onnx"
    mel_path = model_dir / "melspectrogram.onnx" if (model_dir / "melspectrogram.onnx").exists() else get_mel_model_path()
    embedding_path = model_dir / "embedding_model.onnx" if (model_dir / "embedding_model.onnx").exists() else get_embedding_model_path()

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier ONNX model not found: {classifier_path}")

    # Load ONNX sessions with CUDAExecutionProvider on GPU
    mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path)
    mel_frontend._onnx_session = ort.InferenceSession(str(mel_path), providers=providers)

    speech_embedding = SpeechEmbedding(onnx_path=embedding_path)
    speech_embedding._session = ort.InferenceSession(str(embedding_path), providers=providers)

    classifier_session = ort.InferenceSession(str(classifier_path), providers=providers)
    classifier_input_name = classifier_session.get_inputs()[0].name

    logger.info(f"Loaded ONNX sessions on GPU with providers: {classifier_session.get_providers()}")

    # Collect audio files downloaded so far
    audio_files = sorted(list(dataset_dir.glob("**/*.wav")) + list(dataset_dir.glob("**/*.flac")))
    logger.info(f"Found {len(audio_files)} audio files in {dataset_dir}")

    if not audio_files:
        logger.error("No audio files found to evaluate!")
        return

    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    total_samples = 0
    total_segments = 0
    all_scores: list[float] = []
    false_positives_log: list[dict] = []

    start_time = time.time()

    for file_path in tqdm(audio_files, desc="Batched GPU Evaluation (VietSuperSpeech)", unit="file"):
        try:
            audio_np, sr = sf.read(str(file_path), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            if sr != SAMPLE_RATE:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = torch.from_numpy(audio_np).unsqueeze(0)
                audio_np = resamplers[sr](waveform).squeeze(0).numpy()

            total_samples += len(audio_np)

            # 1. Compute Mel Spectrogram for full audio on GPU
            mel = mel_frontend(audio_np[np.newaxis, :])

            # 2. Extract Speech Embeddings (80ms stride) in GPU batches
            embeddings = speech_embedding.extract_embeddings(
                mel, window_size=EMBEDDING_WINDOW, stride=EMBEDDING_STRIDE, batch_size=256
            )

            if embeddings.shape[1] == 0:
                continue

            n_windows = embeddings.shape[1]
            embs = embeddings[0]

            # 3. Construct (n_windows, 16, 96) sequences
            sequences = []
            for i in range(n_windows):
                if i + 1 < N_TIMESTEPS:
                    pad_len = N_TIMESTEPS - (i + 1)
                    seq = np.concatenate([np.zeros((pad_len, 96), dtype=np.float32), embs[: i + 1]], axis=0)
                else:
                    seq = embs[i - N_TIMESTEPS + 1 : i + 1]
                sequences.append(seq)

            batch_seq = np.stack(sequences, axis=0).astype(np.float32)
            total_segments += n_windows

            # 4. Predict Classifier ONNX on GPU
            classifier_outputs = classifier_session.run(None, {classifier_input_name: batch_seq})
            scores = classifier_outputs[0].squeeze(-1).tolist()
            all_scores.extend(scores)

            # 5. Log False Positives with exact timestamp & score for listening back
            rel_path = str(file_path.relative_to(dataset_dir))
            for i, score in enumerate(scores):
                if score >= MIN_LOG_THRESHOLD:
                    start_sec = round((i * EMBEDDING_STRIDE * 160) / SAMPLE_RATE, 2)
                    end_sec = round(start_sec + 2.0, 2)
                    false_positives_log.append({
                        "file": str(file_path),
                        "relative_path": rel_path,
                        "start_time_sec": start_sec,
                        "end_time_sec": end_sec,
                        "score": round(score, 4),
                    })

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    elapsed_time = time.time() - start_time
    total_hours = total_samples / (SAMPLE_RATE * 3600.0)

    logger.info(f"\nEvaluation completed in {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Total audio duration: {total_hours:.4f} hours")
    logger.info(f"Total 80ms streaming windows evaluated: {total_segments}")
    logger.info(f"Total False Positive events logged (score >= 0.5): {len(false_positives_log)}")

    scores_arr = np.array(all_scores) if all_scores else np.array([])

    results_by_thresh = {}
    for t in THRESHOLDS:
        fp_count = int(np.sum(scores_arr >= t)) if len(scores_arr) > 0 else 0
        fpph = fp_count / total_hours if total_hours > 0 else 0.0
        results_by_thresh[str(t)] = {
            "fp_count": fp_count,
            "fpph": round(fpph, 6),
        }
        logger.info(f"Threshold = {t:.2f} -> False Positives: {fp_count}, FPPH: {fpph:.4f}")

    eval_output = {
        "dataset": "thanhnew2001/VietSuperSpeech",
        "checkpoint": str(model_dir),
        "total_segments_evaluated": total_segments,
        "total_hours_evaluated": round(total_hours, 4),
        "execution_provider": "CUDAExecutionProvider",
        "elapsed_seconds": round(elapsed_time, 2),
        "results": results_by_thresh,
    }

    output_json.write_text(json.dumps(eval_output, indent=2) + "\n")
    logger.info(f"Saved evaluation metrics to {output_json}")

    # Save detailed False Positive log with timestamps
    fp_log_json.write_text(json.dumps(false_positives_log, indent=2) + "\n")
    logger.info(f"Saved detailed False Positive log (for audio listening) to {fp_log_json}")


if __name__ == "__main__":
    main()
