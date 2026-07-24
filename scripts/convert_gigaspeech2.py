"""GigaSpeech 2 Dataset Converter for livekit-wakeword.

Converts GigaSpeech 2 Vietnamese (.tar.gz containing .opus/.wav audio) into:
1. 16kHz mono 2.0s WAV clips for audio augmentation / background negatives.
2. Pre-extracted (N, 16, 96) feature matrices (.npy) for direct mmap training.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_gigaspeech2")


def get_onnx_providers() -> list[str]:
    """Auto-detect available ONNX execution providers (CUDA GPU if available, else CPU)."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        logger.info("⚡ Found GPU (CUDAExecutionProvider). Using GPU acceleration for ONNX feature extraction.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        logger.info("💻 Using CPUExecutionProvider for ONNX feature extraction.")
        return ["CPUExecutionProvider"]


def resample_and_mono(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Convert audio to mono and resample to target_sr."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    return audio


def chunk_audio(
    audio: np.ndarray,
    target_samples: int = 32000,
    stride_samples: int = 32000,
    min_samples: int = 8000,
) -> list[np.ndarray]:
    """Slice audio into fixed target_samples (default 2.0s at 16kHz)."""
    n_samples = len(audio)
    if n_samples < min_samples:
        return []

    if n_samples <= target_samples:
        pad_len = target_samples - n_samples
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        padded = np.pad(audio, (pad_left, pad_right), mode="constant")
        return [padded]

    chunks = []
    for start in range(0, n_samples - min_samples, stride_samples):
        end = start + target_samples
        if end <= n_samples:
            chunks.append(audio[start:end])
        else:
            chunk = audio[start:n_samples]
            if len(chunk) >= min_samples:
                padded = np.pad(chunk, (0, target_samples - len(chunk)), mode="constant")
                chunks.append(padded)
            break
    return chunks


def extract_features_batch_gpu(
    chunks: list[np.ndarray],
    mel_sess,
    emb_sess,
    batch_size: int = 256,
) -> list[np.ndarray]:
    """Extract features in parallel GPU batches for maximum VRAM throughput."""
    if not chunks:
        return []

    from livekit.wakeword.data.features import _pad_or_truncate

    mel_in = mel_sess.get_inputs()[0].name
    emb_in = emb_sess.get_inputs()[0].name

    all_embeddings = []
    window_size, stride, n_windows = 76, 8, 16

    for start_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start_idx : start_idx + batch_size]
        audio_batch = np.stack(batch_chunks, axis=0).astype(np.float32)

        mel_raw = mel_sess.run(None, {mel_in: audio_batch})[0]
        if mel_raw.ndim == 4:
            mel_raw = mel_raw[:, 0, :, :]
        mel = mel_raw / 10.0 + 2.0

        b_size = len(batch_chunks)
        windows_list = []
        for b in range(b_size):
            for i in range(n_windows):
                windows_list.append(mel[b, i * stride : i * stride + window_size, :])
        windows_batch = np.stack(windows_list, axis=0)[..., np.newaxis].astype(np.float32)

        emb_out = emb_sess.run(None, {emb_in: windows_batch})[0]
        embeddings = emb_out.squeeze(axis=(1, 2)).reshape(b_size, 16, 96)

        for b in range(b_size):
            all_embeddings.append(_pad_or_truncate(embeddings[b]))

    return all_embeddings


def convert_gigaspeech2(
    input_path: Path,
    output_dir: Path,
    export_wav: bool = False,
    export_features: bool = True,
    max_tar_files: int | None = None,
    max_clips: int | None = None,
    clip_duration: float = 2.0,
    gpu_batch_size: int = 256,
    flush_interval: int = 10000,
    use_local_ssd_cache: bool = True,
) -> None:
    """Find tarball archives under input_path and process them with GPU batching."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    if export_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.name.endswith((".tar.gz", ".tar")):
        tar_files = [input_path]
    elif input_path.is_dir():
        tar_files = sorted(list(input_path.glob("**/*.tar.gz")) + list(input_path.glob("**/*.tar")))
    else:
        raise ValueError(f"Input path {input_path} does not exist or is invalid")

    if not tar_files:
        logger.error(f"No .tar.gz or .tar files found in {input_path}")
        return

    if max_tar_files:
        tar_files = tar_files[:max_tar_files]

    target_samples = int(clip_duration * 16000)
    stride_samples = target_samples
    out_npy = output_dir / "gigaspeech2_vi_features.npy"

    logger.info("==================================================")
    logger.info("🎙️ GigaSpeech 2 Converter (GPU Batching Enabled):")
    logger.info(f"  Input Path:        {input_path}")
    logger.info(f"  Output Directory:  {output_dir}")
    logger.info(f"  Total Archives:    {len(tar_files)}")
    logger.info(f"  GPU Batch Size:    {gpu_batch_size}")
    logger.info(f"  Clip Duration:     {clip_duration:.1f}s ({target_samples} samples)")
    logger.info("==================================================")

    mel_sess, emb_sess = None, None
    if export_features:
        import onnxruntime as ort
        from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

        providers = get_onnx_providers()
        mel_sess = ort.InferenceSession(str(get_mel_model_path()), providers=providers)
        emb_sess = ort.InferenceSession(str(get_embedding_model_path()), providers=providers)

    all_features: list[np.ndarray] = []
    clip_counter = 0

    for idx, drive_tar in enumerate(tqdm(tar_files, desc="Archives")):
        if max_clips and clip_counter >= max_clips:
            break

        active_tar = drive_tar
        local_tar = Path("/tmp") / f"tmp_{drive_tar.name}"
        if use_local_ssd_cache and drive_tar.is_file():
            shutil.copy2(drive_tar, local_tar)
            active_tar = local_tar

        archive_chunks = []
        try:
            with tarfile.open(str(active_tar), "r:*") as tar:
                for member in tar:
                    if max_clips and (clip_counter + len(archive_chunks)) >= max_clips:
                        break
                    fname = member.name.lower()
                    if not (fname.endswith(".opus") or fname.endswith(".wav") or fname.endswith(".flac")):
                        continue
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    try:
                        data, sr = sf.read(io.BytesIO(f.read()))
                    except Exception:
                        continue
                    if len(data) == 0:
                        continue
                    audio_16k = resample_and_mono(data, sr, target_sr=16000)
                    chunks = chunk_audio(audio_16k, target_samples=target_samples, stride_samples=stride_samples)
                    for chunk in chunks:
                        if max_clips and (clip_counter + len(archive_chunks)) >= max_clips:
                            break
                        archive_chunks.append(chunk)
                        if export_wav:
                            sf.write(str(wav_dir / f"clip_{clip_counter + len(archive_chunks):08d}.wav"), chunk, 16000)

            if export_features and archive_chunks and mel_sess and emb_sess:
                feats = extract_features_batch_gpu(archive_chunks, mel_sess, emb_sess, batch_size=gpu_batch_size)
                all_features.extend(feats)

            clip_counter += len(archive_chunks)

        except Exception as e:
            logger.error(f"Error reading tar file {drive_tar.name}: {e}")
        finally:
            if use_local_ssd_cache and local_tar.exists():
                local_tar.unlink()

        if export_features and len(all_features) >= flush_interval:
            chunk_array = np.stack(all_features, axis=0).astype(np.float32)
            if out_npy.exists():
                existing = np.load(str(out_npy))
                combined = np.concatenate([existing, chunk_array], axis=0)
                np.save(str(out_npy), combined)
            else:
                np.save(str(out_npy), chunk_array)
            all_features.clear()
            logger.info(f"Flushed feature batch to {out_npy} (Total saved clips so far: {clip_counter})")

    if export_features and all_features:
        chunk_array = np.stack(all_features, axis=0).astype(np.float32)
        if out_npy.exists():
            existing = np.load(str(out_npy))
            combined = np.concatenate([existing, chunk_array], axis=0)
            np.save(str(out_npy), combined)
        else:
            np.save(str(out_npy), chunk_array)
        all_features.clear()

    logger.info(f"✅ Total 2.0s clips generated: {clip_counter}")
    if export_features and out_npy.exists():
        final_array = np.load(str(out_npy), mmap_mode="r")
        logger.info(f"💾 Feature Matrix File: {out_npy}")
        logger.info(f"  Shape: {final_array.shape}")
        logger.info(f"  File Size: {out_npy.stat().st_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert GigaSpeech 2 archives to livekit-wakeword format.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory containing .tar.gz files or a single .tar.gz file")
    parser.add_argument("--output", "-o", type=str, default="./gigaspeech2_converted", help="Output directory")
    parser.add_argument("--wav", action="store_true", default=False, help="Export 16kHz WAV clips (default: False)")
    parser.add_argument("--features", action="store_true", default=True, help="Extract ONNX (16, 96) .npy features (default: True)")
    parser.add_argument("--max-tars", type=int, default=None, help="Limit maximum number of .tar.gz archives to process")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit maximum number of 2s clips to extract")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size for ONNX inference (default: 256)")

    args = parser.parse_args()
    convert_gigaspeech2(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        export_wav=args.wav,
        export_features=args.features,
        max_tar_files=args.max_tars,
        max_clips=args.max_clips,
        gpu_batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
