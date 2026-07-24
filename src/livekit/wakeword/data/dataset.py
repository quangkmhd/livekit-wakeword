"""Memory-mapped dataset for training with mixed positive/negative batches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class MmapConcat:
    """Concatenates multiple memory-mapped numpy arrays without loading them into RAM."""

    def __init__(self, paths: list[Path]):
        self.arrays: list[np.ndarray] = []
        self.sizes: list[int] = []
        for p in sorted(paths):
            arr = np.load(str(p), mmap_mode="r")
            # Reshape 2D (N, 96) -> 3D (N//16, 16, 96) for pre-extracted embeddings
            if arr.ndim == 2 and arr.shape[1] == 96:
                n_full = (arr.shape[0] // 16) * 16
                arr = arr[:n_full].reshape(-1, 16, 96)
            elif arr.ndim == 3 and arr.shape[2] != 96:
                raise ValueError(
                    f"Feature dimension mismatch: expected 96, got {arr.shape[2]} in file {p}"
                )
            self.arrays.append(arr)
            self.sizes.append(arr.shape[0])

        self.total = sum(self.sizes)
        # We define shape property to mimic numpy array
        self.shape = (self.total, 16, 96)
        self.dtype = self.arrays[0].dtype if self.arrays else np.float32

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: object) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            curr_idx = int(idx)
            if curr_idx < 0:
                curr_idx += self.total
            for arr, size in zip(self.arrays, self.sizes):
                if curr_idx < size:
                    return arr[curr_idx]
                curr_idx -= size
            raise IndexError("Index out of range")

        indices = np.asanyarray(idx)
        if len(indices) == 0:
            return np.empty((0, 16, 96), dtype=self.dtype)

        # Optimize contiguous indexes (e.g. np.arange(pos, pos + n))
        is_contiguous = len(indices) > 1 and np.all(np.diff(indices) == 1)
        if is_contiguous or len(indices) == 1:
            start_idx = int(indices[0])
            if start_idx < 0:
                start_idx += self.total

            curr_idx = start_idx
            for arr, size in zip(self.arrays, self.sizes):
                if curr_idx < size:
                    if curr_idx + len(indices) <= size:
                        return arr[curr_idx:curr_idx + len(indices)]
                    else:
                        # Split across boundary
                        part1_len = size - curr_idx
                        part1 = arr[curr_idx:]
                        rest_indices = indices[part1_len:]
                        part2 = self[rest_indices]
                        return np.concatenate([part1, part2], axis=0)
                curr_idx -= size

        # Fallback for wrapped or non-contiguous indices (like wrap-around % total)
        diffs = np.diff(indices)
        wrap_points = np.where(diffs != 1)[0]
        if len(wrap_points) > 0:
            # Split at wrap points and concatenate
            parts = []
            start = 0
            for wp in wrap_points:
                parts.append(self[indices[start:wp+1]])
                start = wp + 1
            parts.append(self[indices[start:]])
            return np.concatenate(parts, axis=0)

        # General non-contiguous indices
        results = []
        for i in indices:
            curr_idx = int(i)
            if curr_idx < 0:
                curr_idx += self.total
            found = False
            for arr, size in zip(self.arrays, self.sizes):
                if curr_idx < size:
                    results.append(arr[curr_idx])
                    found = True
                    break
                curr_idx -= size
            if not found:
                raise IndexError(f"Index {i} out of range")
        return np.stack(results, axis=0)


def mmap_batch_generator(
    data_files: dict[str, str | Path | list[str | Path]],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate mixed batches from memory-mapped .npy files.

    Each batch contains samples from each class according to n_per_class.
    Files are memory-mapped so data larger than RAM can be used.

    Yields:
        (features, labels) where features is (batch_size, 16, 96)
        and labels is (batch_size,) with 0/1 values.
    """
    # Memory-map all files
    mmaps: dict[str, np.ndarray | MmapConcat] = {}
    for name, paths in data_files.items():
        if isinstance(paths, (str, Path)):
            path = Path(paths)
            if not path.exists():
                logger.warning(f"Data path not found: {path}, skipping class '{name}'")
                continue
            if path.is_dir():
                # Find all .npy files in the directory
                npy_files = list(path.glob("*.npy"))
                if not npy_files:
                    logger.warning(f"No .npy files found in directory {path}, skipping class '{name}'")
                    continue
                mmaps[name] = MmapConcat(npy_files)
                logger.info(f"Loaded {name}: shape={mmaps[name].shape} (concatenated {len(npy_files)} files) from {path}")
                continue
            else:
                data = np.load(str(path), mmap_mode="r")
        elif isinstance(paths, list):
            valid_paths = [Path(p) for p in paths if Path(p).exists()]
            if not valid_paths:
                logger.warning(f"No valid paths found in list for class '{name}', skipping")
                continue
            mmaps[name] = MmapConcat(valid_paths)
            logger.info(f"Loaded {name}: shape={mmaps[name].shape} (concatenated {len(valid_paths)} files)")
            continue
        else:
            raise TypeError(f"Invalid path type for class '{name}': {type(paths)}")

        # Reshape 2D (N, 96) → 3D (N//16, 16, 96) for pre-extracted embeddings
        if data.ndim == 2 and data.shape[1] == 96:
            n_full = (data.shape[0] // 16) * 16
            data = data[:n_full].reshape(-1, 16, 96)
        # Validate embedding dimension matches expected 96-dim vectors
        if data.ndim == 3 and data.shape[2] != 96:
            raise ValueError(
                f"Feature dimension mismatch for '{name}': expected 96, "
                f"got {data.shape[2]}. The file {path} may have been generated "
                f"with a different embedding model."
            )
        mmaps[name] = data
        logger.info(f"Loaded {name}: shape={mmaps[name].shape} from {path}")

    if not mmaps:
        raise FileNotFoundError("No data files found for training")

    # Warn about requested classes that have no loaded data
    for name, n in n_per_class.items():
        if n > 0 and name not in mmaps:
            logger.warning(
                f"Class '{name}' requested {n} samples per batch but no data file was loaded"
            )

    # Track position in each file
    positions: dict[str, int] = {name: 0 for name in mmaps}

    while True:
        batch_features: list[np.ndarray] = []
        batch_labels: list[int] = []

        for name, data in mmaps.items():
            n = n_per_class.get(name, 0)
            if n == 0:
                continue

            label_fn = label_funcs[name]
            total = data.shape[0]
            pos = positions[name]

            # Collect n samples, wrapping around if needed
            indices = np.arange(pos, pos + n) % total
            samples = data[indices]

            for sample in samples:
                batch_features.append(sample)
                batch_labels.append(label_fn(sample))

            positions[name] = (pos + n) % total

        if not batch_features:
            break

        features = np.stack(batch_features, axis=0)
        labels = np.array(batch_labels, dtype=np.float32)

        # Shuffle within batch
        perm = np.random.permutation(len(labels))
        yield features[perm], labels[perm]


class WakeWordDataset(IterableDataset):  # type: ignore[type-arg]
    """IterableDataset wrapping mmap_batch_generator for DataLoader."""

    def __init__(
        self,
        data_files: dict[str, str | Path | list[str | Path]],
        n_per_class: dict[str, int],
        label_funcs: dict[str, Callable[[np.ndarray], int]],
    ):
        self.data_files = data_files
        self.n_per_class = n_per_class
        self.label_funcs = label_funcs

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        gen = mmap_batch_generator(
            data_files=self.data_files,
            n_per_class=self.n_per_class,
            label_funcs=self.label_funcs,
        )
        for features, labels in gen:
            yield (
                torch.from_numpy(features.copy()),
                torch.from_numpy(labels.copy()),
            )


def create_dataloader(
    data_files: dict[str, str | Path | list[str | Path]],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
    prefetch_factor: int = 16,
    num_workers: int = 0,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from memory-mapped feature files."""
    dataset = WakeWordDataset(
        data_files=data_files,
        n_per_class=n_per_class,
        label_funcs=label_funcs,
    )
    return DataLoader(
        dataset,
        batch_size=None,  # Dataset yields pre-batched data
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
