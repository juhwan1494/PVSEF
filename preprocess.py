"""Preprocessing (Methods, "Comparative models for performance evaluation" and "The proposed deep learning framework").

Every frame of each clip is resized to 224 x 224 and normalized with the ViViT image processor.
  baseline : 32 frames sampled at equal intervals from the whole clip (Eq. 1)          -> <out_dir>/<donor_id>.npy
  pvsef    : K = ceil(T/32) consecutive patches of 32 frames (Eq. 5); an incomplete
             last patch is completed with frames from the previous section             -> <out_dir>/<donor_id>/<donor_id>_<k>.npy
Each saved array has shape (32, 3, 224, 224).

Usage:
  python preprocess.py --video_dir <clips> --out_dir <dir> --method baseline
  python preprocess.py --video_dir <clips> --out_dir <dir> --method pvsef
Clips are expected as <video_dir>/<donor_id>.<ext> (one clip per donor)."""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm
from transformers import VivitImageProcessor

from common import PRETRAINED, baseline_indices, patch_indices, read_all_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--method', choices=['baseline', 'pvsef'], required=True)
    ap.add_argument('--ext', default='avi')
    a = ap.parse_args()

    processor = VivitImageProcessor.from_pretrained(PRETRAINED)
    def normalize(frames):
        return processor(list(frames), return_tensors='np')['pixel_values'][0].astype(np.float32)

    videos = sorted(glob.glob(os.path.join(a.video_dir, f'*.{a.ext}')))
    os.makedirs(a.out_dir, exist_ok=True)
    for path in tqdm(videos):
        donor = os.path.splitext(os.path.basename(path))[0]
        frames = read_all_frames(path)
        T = len(frames)
        if a.method == 'baseline':
            np.save(os.path.join(a.out_dir, f'{donor}.npy'), normalize(frames[baseline_indices(T)]))
        else:
            d = os.path.join(a.out_dir, donor)
            os.makedirs(d, exist_ok=True)
            for k, idx in enumerate(patch_indices(T), start=1):
                np.save(os.path.join(d, f'{donor}_{k}.npy'), normalize(frames[idx]))


if __name__ == '__main__':
    main()
