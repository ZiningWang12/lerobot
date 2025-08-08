#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _make_new_features(original_meta: LeRobotDatasetMetadata) -> dict:
    """Build a features dict for the new dataset with:
    - camera keys renamed from observation.images.* -> observation.image.* (singular)
    - same observation.state and action as original
    - adds next.reward and next.done
    The DEFAULT_FEATURES will be automatically appended by LeRobotDatasetMetadata.create.
    """
    features: dict[str, dict] = {}

    # Copy state and action when present
    for key in ("observation.state", "action"):
        if key in original_meta.features:
            features[key] = dict(original_meta.features[key])

    # Add reward/done features expected by downstream training
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}

    # Rename camera keys
    for cam_key in original_meta.camera_keys:
        new_key = cam_key.replace("observation.images", "observation.image")
        original = original_meta.features[cam_key]
        # Ensure dtype is set to video and names present
        features[new_key] = {
            "dtype": original.get("dtype", "video"),
            "shape": tuple(original["shape"]),
            "names": original.get("names", ["channels", "height", "width"]),
        }

    return features


def load_episode_labels(labels_dir: Path, num_episodes: int) -> dict[int, np.ndarray]:
    episode_to_labels: dict[int, np.ndarray] = {}
    for ep in range(num_episodes):
        npy_path = labels_dir / f"episode_{ep}_labels.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {npy_path}")
        arr = np.load(npy_path)
        episode_to_labels[ep] = arr
    return episode_to_labels


def build_labeled_dataset(
    base_repo_id: str,
    labels_dir: Path,
    new_repo_id: str,
    dataset_root: Path | None = None,
    drop_ignore: bool = True,
) -> Path:
    """Create a labeled copy of the dataset with next.reward injected.

    - base_repo_id: original dataset repo id (local copy under HF_LEROBOT_HOME is used)
    - labels_dir: directory containing episode_{i}_labels.npy with values in {-1,0,1}
    - new_repo_id: new dataset repo id to create under the same local cache root
    - drop_ignore: if True, frames with label == -1 are skipped entirely
    """
    logging.info(f"Loading original dataset metadata: {base_repo_id}")
    original_meta = LeRobotDatasetMetadata(base_repo_id, root=dataset_root)

    logging.info("Loading labels")
    episode_to_labels = load_episode_labels(labels_dir, original_meta.total_episodes)

    logging.info("Creating new dataset for writing")
    features = _make_new_features(original_meta)
    writer = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=original_meta.fps,
        features=features,
        root=dataset_root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    # Prepare dataset for frame iteration
    original_ds = LeRobotDataset(base_repo_id, root=dataset_root, download_videos=True)

    prev_ep_idx = None
    kept_frames = 0
    skipped_ignore = 0

    logging.info("Transcoding frames and injecting labels...")
    for idx in tqdm(range(len(original_ds))):
        item = original_ds[idx]
        ep_idx = int(item["episode_index"].item())
        fr_idx = int(item["frame_index"].item())

        # Boundary handling: save episode when episode index changes (and not first)
        if prev_ep_idx is None:
            prev_ep_idx = ep_idx
        elif ep_idx != prev_ep_idx:
            writer.save_episode()
            prev_ep_idx = ep_idx

        labels_arr = episode_to_labels[ep_idx]
        if fr_idx >= labels_arr.shape[0]:
            # If labels are shorter than episode frames (shouldn't happen), skip safely
            skipped_ignore += 1
            continue
        label = int(labels_arr[fr_idx])
        if drop_ignore and label == -1:
            skipped_ignore += 1
            continue
        reward_value = 1.0 if label == 1 else 0.0

        # Build new frame
        new_frame = {}
        # cameras: rename keys
        for cam_key in original_meta.camera_keys:
            new_key = cam_key.replace("observation.images", "observation.image")
            img = item[cam_key]
            # Ensure numpy and HWC
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
                img = img.transpose(1, 2, 0)
            new_frame[new_key] = img
        # copy optional state/action if available
        if "observation.state" in original_meta.features:
            state = item["observation.state"]
            if hasattr(state, "numpy"):
                state = state.numpy()
            new_frame["observation.state"] = state
        if "action" in original_meta.features:
            act = item["action"]
            if hasattr(act, "numpy"):
                act = act.numpy()
            new_frame["action"] = act
        # labels
        new_frame["next.reward"] = np.array([reward_value], dtype=np.float32)
        new_frame["next.done"] = np.array([False])

        task = item.get("task", "")
        writer.add_frame(new_frame, task=task)
        kept_frames += 1

    writer.save_episode()

    logging.info(
        f"Done. Kept frames: {kept_frames}, skipped ignore: {skipped_ignore}. New dataset at: {writer.root}"
    )
    return writer.root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a labeled LeRobot dataset with next.reward from .npy labels")
    p.add_argument("--base_repo_id", type=str, required=True, help="Original dataset repo id, e.g. wzn12/teleop_ring")
    p.add_argument(
        "--labels_dir",
        type=Path,
        required=True,
        help="Directory containing episode_{i}_labels.npy files (and optionally labeled_data.pt)",
    )
    p.add_argument(
        "--new_repo_id",
        type=str,
        default=None,
        help="New dataset repo id (defaults to '<user>/<dataset>_labeled')",
    )
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Override HF_LEROBOT_HOME root. Usually leave unset to use default cache dir.",
    )
    p.add_argument("--keep_ignore", action="store_true", help="Keep frames with label -1 as reward=0")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    new_repo_id = args.new_repo_id
    if new_repo_id is None:
        if "/" in args.base_repo_id:
            user, name = args.base_repo_id.split("/", 1)
        else:
            user, name = "local", args.base_repo_id
        new_repo_id = f"{user}/{name}_labeled"

    build_labeled_dataset(
        base_repo_id=args.base_repo_id,
        labels_dir=args.labels_dir,
        new_repo_id=new_repo_id,
        dataset_root=args.dataset_root,
        drop_ignore=not args.keep_ignore,
    )


if __name__ == "__main__":
    main()