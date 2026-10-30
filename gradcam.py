"""Out-of-fold Grad-CAM of PVSEF (Results, "Visualization of model attribution with Grad-CAM"; Fig. 4, Supplementary Fig. S1).

Grad-CAM is computed for the positive-class logit. Gradients and activations are taken at the layer normalization that
precedes self-attention in the last encoder block of ViViT; the 3,136 tubelet tokens are arranged in their native
16 (temporal) x 14 x 14 (spatial) grid (each temporal slice covers two consecutive frames). Channel weights are the
token-averaged gradients, and the map is the ReLU of the weighted sum of activations, normalized to the maximum of the
32-frame input. Each test donor of a fold is explained by the model of that fold (out-of-fold).

Frames shown in the paper are illustrative. Candidate frames (frame maximum >= 0.5 of the input maximum) of correctly
classified donors are pre-ranked with colour masks, separately for activation on the biopsy cut surface / fresh blood
and for activation on the intact liver surface; the displayed frames are then chosen by visual review.

Outputs (in --out_dir):
  cam/<donor_id>_<k>.npz      Grad-CAM volume (16 x 14 x 14) and positive-class logit of every patch
  frame_scores.csv            one row per temporal slice: peak, share of activation on the liver / blood masks, ...
  review/cut_surface/*.png, review/liver/*.png   frame + overlay of the top candidates (--top) for visual review

Usage:
  python gradcam.py --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir data_pvsef \
                    --checkpoint ckpt/pvsef_Macro_fold1.pth --scores_csv scores/pvsef_Macro_fold1.csv --out_dir cam_Macro"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import ViViTClassifier, donor_inputs, read_folds
from evaluate import closest_to_01


def to_rgb(x):
    """Inverse of the ViViT image processor for one frame (3, 224, 224) -> uint8 RGB (224, 224, 3)."""
    return np.clip(((x * 0.5 + 0.5) + 1) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def tissue_masks(rgb):
    """Frame-adaptive colour masks (OpenCV HSV, H 0-179), used only to pre-rank candidate frames for visual review."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    red = (h <= 8) | (h >= 172)
    metal = (s < 35) | ((s < 60) & (v > 200))                          # instruments, clips, smoke
    specular = v > 245
    gallbladder = (h >= 95) & (h <= 140) & (v > 140)                   # gallbladder / white-blue peritoneum
    fat = (h >= 10) & (h <= 38) & (s >= 80)
    blood = red & (s >= max(150, int(np.median(s)) + 70)) & (v >= 90)  # fresh blood / biopsy cut surface
    liver = ((h >= 135) | (h <= 10)) & ~(metal | specular | gallbladder | fat | blood) & (v > 40)
    return dict(liver=liver, blood=blood)


class ViViTGradCAM:
    def __init__(self, net):
        self.net = net
        net.model.vivit.encoder.layer[-1].layernorm_before.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.act = output
        output.register_hook(lambda g: setattr(self, 'grad', g))

    def __call__(self, x, target=1):
        self.net.zero_grad()
        z = self.net(x)                                                 # two logits
        z[0, target].backward()
        A, G = self.act[0, 1:].detach(), self.grad[0, 1:].detach()     # tubelet tokens (3136, 768), CLS token excluded
        cam = F.relu((A * G.mean(0)).sum(-1)).reshape(16, 14, 14).cpu().numpy()
        return cam / (cam.max() + 1e-8), float(z[0, 1].detach())


def overlay(rgb, cam2d, alpha=0.4):
    c = cv2.resize(cam2d.astype(np.float32), rgb.shape[:2][::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
    hm = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * c), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (alpha * hm + (1 - alpha) * rgb).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', choices=['Macro', 'Micro'], required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--folds_csv', required=True)
    ap.add_argument('--data_dir', required=True, help='PVSEF patches written by preprocess.py --method pvsef')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scores_csv', required=True, help='output of test.py for the same fold (training-derived threshold)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--top', type=int, default=24, help='candidate frames per category written for visual review')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ViViTClassifier().to(device)
    net.load_state_dict(torch.load(a.checkpoint, map_location=device)['model_state_dict'])
    net.eval()
    cammer = ViViTGradCAM(net)

    scores = pd.read_csv(a.scores_csv, dtype={'donor_id': str})
    train = scores[scores.split == 'train']
    tau = closest_to_01(train.label.values, train.score.values)            # training-derived threshold (Eq. 10)
    test_scores = scores[scores.split == 'test'].set_index('donor_id').score

    df = read_folds(a.folds_csv, a.endpoint)
    df = df[df.fold == a.fold]
    os.makedirs(os.path.join(a.out_dir, 'cam'), exist_ok=True)
    rows = []
    for donor, label in zip(df.donor_id, df.label):
        correct = int((test_scores[donor] >= tau) == bool(label))
        for path in donor_inputs(a.data_dir, donor, 'pvsef'):
            v = np.load(path)
            cam, logit = cammer(torch.from_numpy(v).float().unsqueeze(0).to(device))
            pid = os.path.splitext(os.path.basename(path))[0]
            np.savez_compressed(os.path.join(a.out_dir, 'cam', f'{pid}.npz'), cam=cam.astype(np.float16), logit1=logit)
            for t in range(16):
                rgb = to_rgb(v[2 * t])
                c = cv2.resize(cam[t], (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 1)
                m = tissue_masks(rgb)
                hot = c >= 0.5 * c.max()
                rows.append(dict(donor_id=donor, label=int(label), correct=correct, patch=pid, path=path, t=t,
                                 peak=float(cam[t].max()),
                                 cam_on_blood=float((c * m['blood']).sum() / (c.sum() + 1e-8)),
                                 hot_on_liver=float((hot & m['liver']).sum() / max(hot.sum(), 1)),
                                 blood_area=float(m['blood'].mean())))
        print(a.endpoint, a.fold, donor, flush=True)
    fs = pd.DataFrame(rows)
    fs.to_csv(os.path.join(a.out_dir, 'frame_scores.csv'), index=False)

    # pre-ranking of candidate frames (correctly classified donors with steatosis >= 5%, frame maximum >= 0.5)
    cand = fs[(fs.correct == 1) & (fs.label == 1) & (fs.peak >= 0.5)]
    groups = {
        'cut_surface': cand[(cand.blood_area >= 0.01) & (cand.cam_on_blood >= 0.2)].assign(rank=lambda d: d.cam_on_blood * d.peak),
        'liver': cand[(cand.blood_area < 0.003) & (cand.hot_on_liver >= 0.8)].assign(rank=lambda d: d.hot_on_liver * d.peak),
    }
    for name, g in groups.items():
        g = g.sort_values('rank', ascending=False).drop_duplicates('donor_id').head(a.top)
        d = os.path.join(a.out_dir, 'review', name)
        os.makedirs(d, exist_ok=True)
        for r in g.itertuples():
            v = np.load(r.path)
            cam = np.load(os.path.join(a.out_dir, 'cam', f'{r.patch}.npz'))['cam'].astype(np.float32)
            rgb = to_rgb(v[2 * r.t])
            tile = np.concatenate([rgb, overlay(rgb, cam[r.t])], 1)
            cv2.imwrite(os.path.join(d, f'{r.patch}_t{r.t:02d}.png'), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        print(f'{name}: {len(g)} candidate frames written for visual review')


if __name__ == '__main__':
    main()
