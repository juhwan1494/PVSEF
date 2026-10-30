"""Shared components: video reading, frame sampling (Methods, Eqs. 1 and 5), dataset and network."""
import glob
import os

import av
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import VivitForVideoClassification

CLIP_LEN = 32                                   # frames per network input
FRAME_SIZE = (224, 224)                         # all frames are resized to 224 x 224
PRETRAINED = 'google/vivit-b-16x2-kinetics400'  # ViViT pre-trained on Kinetics-400


def read_all_frames(path):
    """Decode every frame of a clip and resize it to 224 x 224 (RGB, uint8). Returns an array of shape (T, 224, 224, 3)."""
    with av.open(path) as container:
        frames = [cv2.resize(f.to_ndarray(format='rgb24'), FRAME_SIZE) for f in container.decode(video=0)]
    if not frames:
        raise ValueError(f'no decodable frames in {path}')
    return np.stack(frames)


def baseline_indices(T):
    """Conventional method (Eq. 1): 32 frames sampled at equal intervals, one frame every T/32 frames."""
    return np.floor(np.arange(CLIP_LEN) * T / CLIP_LEN).astype(np.int64)


def patch_indices(T):
    """PVSEF (Eq. 5): the clip is divided into K = ceil(T/32) consecutive patches of 32 frames.
    If the last patch contains fewer than 32 frames, frames from the previous section are added to complete it."""
    if T < CLIP_LEN:                                         # not expected for the clips of this study (>= 288 frames)
        return [np.resize(np.arange(T), CLIP_LEN)]
    K = -(-T // CLIP_LEN)
    patches = []
    for k in range(K):
        start = k * CLIP_LEN
        if start + CLIP_LEN > T:                             # last, incomplete patch
            start = T - CLIP_LEN
        patches.append(np.arange(start, start + CLIP_LEN))
    return patches


def read_folds(folds_csv, endpoint):
    """Donor-level fold assignment: CSV with columns donor_id, endpoint ('Macro' or 'Micro'), label (0: <5%, 1: >=5%)
    and fold (1-5; 0 = never in a test fold). All inputs of a donor share the donor's fold."""
    df = pd.read_csv(folds_csv, dtype={'donor_id': str})
    return df[df.endpoint == endpoint].reset_index(drop=True)


def donor_inputs(data_dir, donor_id, method):
    """Paths of the preprocessed inputs of a donor: one file for the baseline, K patch files for PVSEF."""
    if method == 'baseline':
        return [os.path.join(data_dir, f'{donor_id}.npy')]
    files = glob.glob(os.path.join(data_dir, donor_id, f'{donor_id}_*.npy'))
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0].rsplit('_', 1)[1]))


class VideoDataset(Dataset):
    """Each 32-frame input is an independent sample carrying the label of its donor (Eq. 7 for PVSEF)."""

    def __init__(self, items):
        self.items = items                                  # list of (path, label, donor_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, donor = self.items[idx]
        return torch.from_numpy(np.load(path)).float(), torch.tensor(label, dtype=torch.long), donor


def build_items(df, data_dir, method):
    items = []
    for donor, label in zip(df.donor_id, df.label):
        for path in donor_inputs(data_dir, donor, method):
            items.append((path, int(label), donor))
    return items


class ViViTClassifier(nn.Module):
    """ViViT backbone initialised from Kinetics-400 with a two-class output layer.
    The network output z_theta(X) (two logits) is returned; softmax gives f_theta(X) of Eqs. 2 and 6."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.model = VivitForVideoClassification.from_pretrained(PRETRAINED)
        self.fc = nn.Linear(self.model.config.num_labels, n_classes)

    def forward(self, x):
        return self.fc(self.model(x).logits)
