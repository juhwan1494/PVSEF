# PVSEF: patch-based video segmentation and ensemble framework

Code for preprocessing, training and testing accompanying the manuscript
"Prediction of hepatic steatosis from intraoperative laparoscopic videos in living liver donors using a patch-based ensemble deep learning framework".

The single-clip ViViT (baseline) and PVSEF classify macrovesicular and microvesicular steatosis (frozen section, ≥5% vs <5%) from one laparoscopic video clip per donor. The video data and trained model weights are not included (see the Data availability statement of the manuscript).

## Files

| File | Content | Manuscript |
|---|---|---|
| `common.py` | frame reading, frame sampling, dataset, network | Eqs. 1, 2, 5, 6 |
| `preprocess.py` | 224 × 224 frames normalized with the ViViT image processor; baseline input or PVSEF patches | Comparative models; PVSEF; Experimental settings |
| `train.py` | training of one fold | Eqs. 3 and 7; Experimental settings |
| `test.py` | donor-level scores (positive-class logit; mean over the patches for PVSEF) | Eqs. 4, 8, 9 |
| `evaluate.py` | AUC, thresholds, metrics, fold-stratified bootstrap CIs, pooled confusion matrices | Eq. 10; Performance evaluation metrics; Statistical analysis |
| `gradcam.py` | out-of-fold Grad-CAM of PVSEF and pre-ranking of candidate frames for visual review | Results, Grad-CAM; Fig. 4; Supplementary Fig. S1 |

## Method summary

- **Baseline:** 32 frames sampled at equal intervals (one frame every T/32 frames) from the whole clip of T frames.
- **PVSEF:** the clip is divided into K = ⌈T/32⌉ consecutive patches of 32 frames; if the last patch contains fewer than 32 frames, frames from the previous section are added to complete it. During training, each patch is an independent sample with the label of its donor. At inference, the donor-level score is the mean of the positive-class logits of the K patches.
- **Network:** ViViT (`google/vivit-b-16x2-kinetics400`, pre-trained on Kinetics-400) with a two-class output layer.
- **Training (both methods):** batch size 16, 50 epochs, cross-entropy loss, SGD with an initial learning rate of 0.01 reduced by a factor of 10 after the 10th epoch.
- **Validation:** donor-level five-fold cross-validation stratified by outcome class; all inputs of a donor stay in the same fold.
- **Thresholds:** the point on the ROC curve closest to (0, 1), determined on the training donors of each fold and applied unchanged to the test fold (training-derived threshold), or determined on the test fold itself (post-hoc operating point).
- **Grad-CAM:** computed for the positive-class logit at the layer normalization preceding self-attention in the last encoder block of ViViT, with the tubelet tokens arranged in the 16 (temporal) × 14 × 14 (spatial) grid; each donor is explained by the model of the fold in which it was a test case. Candidate frames (frame maximum ≥ 0.5 of the input maximum) of correctly classified donors are pre-ranked with colour masks for the biopsy cut surface / fresh blood and for the intact liver surface, and the displayed frames are chosen by visual review.

## Usage

```bash
# 1. preprocessing (clips as <video_dir>/<donor_id>.avi)
python preprocess.py --video_dir clips --out_dir data_baseline --method baseline
python preprocess.py --video_dir clips --out_dir data_pvsef    --method pvsef

# 2. fold assignment: folds.csv with columns donor_id, endpoint (Macro/Micro), label (0: <5%, 1: >=5%), fold (1-5; 0 = never tested)

# 3. training and testing, for each method, endpoint and fold
python train.py --method pvsef --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir data_pvsef --out_dir ckpt
python test.py  --method pvsef --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir data_pvsef \
                --checkpoint ckpt/pvsef_Macro_fold1.pth --out_csv scores/pvsef_Macro_fold1.csv

# 4. evaluation (needs scores of both methods for folds 1-5)
python evaluate.py --scores_dir scores --endpoint Macro

# 5. Grad-CAM of PVSEF, for each endpoint and fold
python gradcam.py --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir data_pvsef \
                  --checkpoint ckpt/pvsef_Macro_fold1.pth --scores_csv scores/pvsef_Macro_fold1.csv --out_dir cam_Macro_fold1
```

## Environment

Python 3.8; see `requirements.txt`. Experiments were run with PyTorch on NVIDIA RTX A6000 GPUs.
