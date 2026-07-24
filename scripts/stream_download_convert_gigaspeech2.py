"""Stream Download & Convert GigaSpeech 2 Dataset to .npy Feature Shards.

Parallel Producer-Consumer Pipeline:
- Thread 1 (Producer): Pre-downloads the NEXT archive in the background.
- Thread 2 (Consumer - GPU): Converts the CURRENT archive to .npy on GPU and deletes it.

Double buffering + RAM optimization:
- Stores features as contiguous 3D float16 numpy arrays (eliminates Python object overhead).
- Runs garbage collection after each archive to keep memory usage under ~700MB.
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import os
import queue
import re
import tarfile
import threading
from pathlib import Path

# Enable Rust multi-threaded download acceleration for Hugging Face CDN (saturates line rate)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
import onnxruntime as ort
import scipy.signal
import soundfile as sf
from tqdm import tqdm

from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_gigaspeech2")


def get_onnx_providers() -> list[str]:
    """Detect ONNX execution providers (CUDA GPU if available, else CPU)."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        logger.info("⚡ Using CUDAExecutionProvider (GPU acceleration).")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        logger.info("💻 Using CPUExecutionProvider.")
        return ["CPUExecutionProvider"]


def resample_and_mono(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Convert audio to mono float32 and resample to target_sr."""
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
    """Slice audio into fixed target_samples (2.0s at 16kHz)."""
    n_samples = len(audio)
    if n_samples < min_samples:
        return []

    if n_samples <= target_samples:
        pad_len = target_samples - n_samples
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        return [np.pad(audio, (pad_left, pad_right), mode="constant")]

    chunks = []
    for start in range(0, n_samples - min_samples, stride_samples):
        end = start + target_samples
        if end <= n_samples:
            chunks.append(audio[start:end])
        else:
            chunk = audio[start:n_samples]
            if len(chunk) >= min_samples:
                chunks.append(np.pad(chunk, (0, target_samples - len(chunk)), mode="constant"))
            break
    return chunks


def extract_features_batch_gpu(
    chunks: list[np.ndarray],
    mel_sess: ort.InferenceSession,
    emb_sess: ort.InferenceSession,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract (N, 16, 96) float16 embeddings using ONNX Mel + Speech Embedding sessions."""
    if not chunks:
        return np.empty((0, 16, 96), dtype=np.float16)

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

    return np.stack(all_embeddings, axis=0).astype(np.float16)


def natural_sort_key(s: str) -> list[int | str]:
    """Sort strings with embedded numbers naturally (0, 1, 2... 10... 239)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def get_hf_tar_list(lang: str = "vi", split: str = "train") -> list[str]:
    """List relative tar.gz filenames from Hugging Face speechcolab/gigaspeech2 dataset."""
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        pattern = f"datasets/speechcolab/gigaspeech2/data/{lang}/{split}/*.tar.gz"
        full_paths = fs.glob(pattern)
        prefix = "datasets/speechcolab/gigaspeech2/"
        rel_files = [p[len(prefix) :] if p.startswith(prefix) else p for p in full_paths]
        rel_files = sorted(rel_files, key=natural_sort_key)
        logger.info(f"Found {len(rel_files)} archive files on Hugging Face CDN.")
        return rel_files
    except Exception as e:
        logger.error(f"Error fetching HuggingFace file list: {e}")
        return []


def download_hf_file(rel_path: str, tmp_dir: Path) -> Path:
    """Download a single tar.gz file from Hugging Face CDN using hf_transfer multi-threading."""
    from huggingface_hub import hf_hub_download

    logger.info(f"⚡ 🚀 Downloading {rel_path} with Rust multi-threading (hf_transfer)...")
    downloaded_path = hf_hub_download(
        repo_id="speechcolab/gigaspeech2",
        filename=rel_path,
        repo_type="dataset",
        local_dir=str(tmp_dir),
    )
    return Path(downloaded_path)


def process_archive_file(
    tar_path: Path,
    mel_sess: ort.InferenceSession,
    emb_sess: ort.InferenceSession,
    sub_batch_size: int = 256,
    clip_duration: float = 2.0,
) -> list[np.ndarray]:
    """Extract 2s audio chunks from tar file and run ONNX feature extraction."""
    target_samples = int(clip_duration * 16000)
    stride_samples = target_samples
    sub_batch_audio: list[np.ndarray] = []
    features_list: list[np.ndarray] = []

    with tarfile.open(str(tar_path), "r:*") as tar:
        for member in tar:
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
            chunks = chunk_audio(
                audio_16k, target_samples=target_samples, stride_samples=stride_samples
            )

            for chunk in chunks:
                sub_batch_audio.append(chunk)
                if len(sub_batch_audio) >= sub_batch_size:
                    feats = extract_features_batch_gpu(
                        sub_batch_audio, mel_sess, emb_sess, batch_size=sub_batch_size
                    )
                    if len(feats) > 0:
                        features_list.append(feats)
                    sub_batch_audio.clear()

    if sub_batch_audio:
        feats = extract_features_batch_gpu(
            sub_batch_audio, mel_sess, emb_sess, batch_size=sub_batch_size
        )
        if len(feats) > 0:
            features_list.append(feats)
        sub_batch_audio.clear()

    return features_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel RAM-optimized feature extraction for GigaSpeech 2."
    )
    parser.add_argument(
        "--source",
        choices=["hf", "gdrive_gdown", "local"],
        default="hf",
        help="Download source (hf=Hugging Face CDN, gdrive_gdown=gdown Google Drive)",
    )
    parser.add_argument(
        "--gdrive-folder",
        type=str,
        default="1IrYGRBYAMVRCFNLyo70QaYx-SMonMOtZ",
        help="Google Drive Folder ID/URL",
    )
    parser.add_argument(
        "--local-dir", type=str, default="", help="Local directory containing tar.gz files"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./gigaspeech2_converted", help="Output directory"
    )
    parser.add_argument(
        "--tmp-dir", type=str, default="./tmp_download", help="Temp download directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="GPU batch size for feature extraction"
    )
    parser.add_argument(
        "--clips-per-shard",
        type=int,
        default=200000,
        help="Clips per output .npy shard (~1.2GB Float16)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    progress_log = output_dir / "completed_tars.txt"
    completed_tars = set()
    if progress_log.exists():
        with open(progress_log, encoding="utf-8") as f:
            completed_tars = set(line.strip() for line in f if line.strip())

    providers = get_onnx_providers()
    mel_sess = ort.InferenceSession(str(get_mel_model_path()), providers=providers)
    emb_sess = ort.InferenceSession(str(get_embedding_model_path()), providers=providers)

    existing_shards = list(output_dir.glob("gigaspeech2_vi_part*.npy"))
    current_shard_idx = len(existing_shards) + 1
    shard_features: list[np.ndarray] = []
    total_processed_clips = 0

    if args.source == "hf":
        logger.info("🚀 Using Hugging Face CDN (hf_transfer multi-threaded enabled).")
        archive_list = get_hf_tar_list(lang="vi", split="train")
    elif args.source == "local":
        local_path = Path(args.local_dir)
        archive_list = [
            str(p)
            for p in sorted(list(local_path.glob("*.tar.gz")) + list(local_path.glob("*.tar")))
        ]
    else:
        logger.info(f"📥 Using gdown to fetch Google Drive folder: {args.gdrive_folder}")
        import gdown

        gdown.download_folder(
            id=args.gdrive_folder, output=str(tmp_dir), quiet=False, use_cookies=False
        )
        archive_list = [str(p) for p in sorted(tmp_dir.glob("*.tar.gz"))]

    remaining_archives = [a for a in archive_list if Path(a).name not in completed_tars]
    logger.info(
        f"Total Archives: {len(archive_list)} | Completed: {len(completed_tars)} | "
        f"Remaining: {len(remaining_archives)}"
    )

    if not remaining_archives:
        logger.info("🎉 All archives have already been converted!")
        return

    # Queue maxsize=1: double buffering (max 1 file waiting in queue + 1 file currently converting)
    download_queue: queue.Queue[tuple[str, Path] | None] = queue.Queue(maxsize=1)

    def downloader_worker() -> None:
        """Background thread that pre-downloads the next archive file."""
        for archive_ref in remaining_archives:
            archive_name = Path(archive_ref).name
            try:
                if args.source == "hf":
                    local_tar_path = download_hf_file(archive_ref, tmp_dir)
                else:
                    local_tar_path = Path(archive_ref)

                download_queue.put((archive_name, local_tar_path))
            except Exception as err:
                logger.error(f"Failed to download {archive_name}: {err}")

        download_queue.put(None)

    downloader_thread = threading.Thread(target=downloader_worker, daemon=True)
    downloader_thread.start()

    logger.info("⚡ Parallel RAM-Optimized Pipeline Started (Downloader & GPU Converter).")

    pbar = tqdm(total=len(remaining_archives), desc="Archives Converted")
    while True:
        item = download_queue.get()
        if item is None:
            break

        archive_name, local_tar_path = item

        try:
            if not local_tar_path.exists():
                logger.error(f"File {archive_name} not found, skipping.")
                pbar.update(1)
                download_queue.task_done()
                continue

            logger.info(f"🧠 Processing {archive_name} on GPU...")
            feats_list = process_archive_file(
                tar_path=local_tar_path,
                mel_sess=mel_sess,
                emb_sess=emb_sess,
                sub_batch_size=args.batch_size,
            )
            archive_clip_count = sum(arr.shape[0] for arr in feats_list)
            shard_features.extend(feats_list)
            total_processed_clips += archive_clip_count
            logger.info(f"Extracted {archive_clip_count} clips from {archive_name}.")

            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(f"{archive_name}\n")

        except Exception as e:
            logger.error(f"Error processing archive {archive_name}: {e}")
        finally:
            if args.source in ["hf", "gdrive_gdown"] and local_tar_path and local_tar_path.exists():
                local_tar_path.unlink()
                logger.info(f"🗑️ Deleted temp archive {archive_name} to free disk space.")

            gc.collect()
            pbar.update(1)
            download_queue.task_done()

        current_shard_clips = sum(arr.shape[0] for arr in shard_features)
        if current_shard_clips >= args.clips_per_shard:
            combined = np.concatenate(shard_features, axis=0)
            shard_arr = combined[: args.clips_per_shard].astype(np.float16)
            overflow = combined[args.clips_per_shard :]

            shard_file = output_dir / f"gigaspeech2_vi_part{current_shard_idx:02d}.npy"
            np.save(str(shard_file), shard_arr)
            logger.info(
                f"💾 Saved Shard: {shard_file.name} (Clips: {shard_arr.shape[0]}, "
                f"Size: {shard_file.stat().st_size / 1e9:.2f} GB)"
            )
            current_shard_idx += 1

            shard_features = [overflow] if len(overflow) > 0 else []
            del combined, shard_arr
            gc.collect()

    if shard_features:
        combined = np.concatenate(shard_features, axis=0)
        shard_arr = combined.astype(np.float16)
        shard_file = output_dir / f"gigaspeech2_vi_part{current_shard_idx:02d}.npy"
        np.save(str(shard_file), shard_arr)
        logger.info(
            f"💾 Saved Final Shard: {shard_file.name} (Clips: {shard_arr.shape[0]}, "
            f"Size: {shard_file.stat().st_size / 1e9:.2f} GB)"
        )
        del combined, shard_arr, shard_features
        gc.collect()

    downloader_thread.join()
    logger.info(f"🎉 Complete! Processed total 2.0s clips: {total_processed_clips}")


if __name__ == "__main__":
    main()
