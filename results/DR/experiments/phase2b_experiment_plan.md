# FedMorphis Phase 2B: Controlled Improvement Plan

**Author:** DeepMind Antigravity AI Assistant  
**Date:** 2026-10-18  
**Experiment Series:** DR Local Binary Lesion Segmentation Improvement (`results/DR/experiments/`)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, Compute Capability 8.6)  
**Baseline Anchor:** Epoch 16 Checkpoint (`results/DR/checkpoints/best_dr_unet.pth`, Val Dice: 0.3609, Test Dice: 0.2769)  

---

## 1. Technical Audit of Existing System

| Component | Current Implementation in Baseline | Measured Limitations & Diagnostic Findings |
| :--- | :--- | :--- |
| **Dataset Loader** (`src/dr_segmentation_dataset.py`) | Direct bilinear resizing of images to $512 \times 512$; nearest-neighbor resizing of masks to $512 \times 512$. Zero data augmentation. | **Measured:** Non-isotropic squashing distorts aspect ratios up to 1.51 by 33.8%. Nearest-neighbor downsampling erases 5.18% of all lesion components and 9.47% of microaneurysms. |
| **Model Architecture** (`src/shared_unet.py`) | `SharedUNet`: Standard 5-stage U-Net (DoubleConv + MaxPool, ConvTranspose2d + DoubleConv). Total parameters: 31,037,633. | Fits cleanly within 4 GB VRAM at $512 \times 512$ with batch size 2 (~1.8–2.2 GB VRAM peak). Full-image $1024 \times 1024$ would exceed 5.5 GB VRAM and trigger CUDA OOM. |
| **Loss Function** (`src/train_dr_segmentation.py`) | `combined_loss = bce + dice_loss` (1:1 equal unweighted sum). | **Measured:** Lesion pixels are 0.322% vs 99.678% background (~310:1 imbalance). Unweighted BCE gradients overwhelmingly penalize false positives, driving 65.9% of GT pixels to $p < 0.01$ and causing a 1.85× volume under-prediction. |
| **Metric Code** (`src/evaluate_dr_segmentation.py`) | Global dataset-level micro-pooled metrics across all concatenated images. Empty masks yield $\text{Sens}=1.0$. | **Measured:** Global pooled Dice is 0.3609, whereas Macro-averaged (per-image mean) Dice is 0.3191. In DDR validation, 100% of images contain lesions (0 empty masks). |
| **Thresholding** | Hardcoded at $\tau = 0.50$. | **Measured:** Sweeping $\tau$ to 0.05 only raises sensitivity by +3.8% while collapsing precision, because 91.3% of false negative pixels have $p < 0.01$. Post-hoc thresholding cannot fix feature-level suppression. |

---

## 2. Hardware Budget & VRAM Feasibility Analysis (RTX 3050 Laptop, 4 GB)

The local training environment is restricted to an **RTX 3050 Laptop GPU with 4,096 MB VRAM**.

### VRAM Allocation Model:
- **Model Parameters (FP32):** $31.04\text{ M} \times 4\text{ bytes} \approx 124.1\text{ MB}$.
- **Gradients (FP32):** $\approx 124.1\text{ MB}$.
- **Adam Optimizer States:** 2 state tensors (momentum + variance) $\approx 248.2\text{ MB}$.
- **PyTorch / CUDA Runtime Overhead:** $\approx 500 - 600\text{ MB}$.
- **Static Base Memory:** $\approx 1.1\text{ GB}$.
- **Dynamic Forward-Backward Activations:**
  - Shape $[2, 3, 512, 512]$: $\approx 850 - 1,100\text{ MB}$ $\rightarrow$ **Total Peak VRAM $\approx 2.0 - 2.2\text{ GB}$ (Safe, 45% headroom)**.
  - Shape $[4, 3, 512, 512]$: $\approx 1,700 - 2,200\text{ MB}$ $\rightarrow$ **Total Peak VRAM $\approx 3.2 - 3.5\text{ GB}$ (Borderline, risk of OOM on Windows DWM)**.
  - Shape $[1, 3, 1024, 1024]$: Activations scale $4\times$ $\rightarrow$ **Total Peak VRAM $\approx 4.5 - 5.2\text{ GB}$ (Guaranteed CUDA OOM)**.

### Phase 2B Hardware Boundary Decisions:
1. **Resolution:** Fixed to $512 \times 512$ canvas (via letterbox padding or area-preserving downsampling).
2. **Batch Size:** Fixed to `batch_size = 2`. If larger effective batch size is needed, gradient accumulation (`gradient_accumulation_steps = 2`) will be used to simulate `batch_size = 4` with zero additional activation memory.
3. **Precision:** FP32 standard (with optional PyTorch native `torch.amp.autocast('cuda')` if additional headroom is needed).

---

## 3. Strict Experimental Methodology & Protocols

To guarantee scientific validity and prevent data leakage:
1. **One-Factor-at-a-Time (OFAT):** Every experiment modifies exactly one isolated variable relative to its immediate predecessor or the control baseline.
2. **Locked Test Set:** The test split (`data/DR/DDR-dataset/lesion_segmentation/test/`) will remain **completely untouched and unloaded** during all exploratory Phase 2B experiments.
3. **Pre-Registered Decision Metric:**
   - **Primary Metric:** Validation Global Pooled Dice ($\tau = 0.50$).
   - **Secondary Diagnostic Metrics:** Validation Macro Dice, Validation Sensitivity, Pred/GT Area Ratio, Precision, Specificity.
   - **Model Selection Rule:** For each experiment, the checkpoint from the epoch achieving the highest Validation Global Dice is preserved as `best_model.pth`.
4. **Isolated Directory Tree:** Each experiment runs in its own subfolder under `results/DR/experiments/`:
   ```text
   results/DR/experiments/
   ├── exp01_letterbox_padding/
   ├── exp02_area_preserving_masks/
   ├── exp03_pos_weight_5/
   ├── exp04_pos_weight_10/
   ├── exp05_focal_dice/
   ├── exp06_tversky_loss/
   ├── exp07_safe_augmentation/
   └── exp08_best_combination/
   ```
5. **Preservation of Baseline:** Checkpoints and results in `results/DR/checkpoints/`, `results/DR/dr_local_test_metrics.txt`, and `results/DR/validation_predictions/` are treated as read-only anchors.

---

## 4. Phase 2B Controlled Experiment Matrix

| Exp ID | Experiment Name | Hypothesis | Changed Variable | Fixed Variables | Evaluation Criteria | Expected Cost (VRAM / Time) | Potential Risks |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **EXP-0** *(Control)* | Original Baseline Anchor | Reference anchor from Phase 2. | None | Resolution 512×512, squashed, unweighted BCE+Dice, bs=2, lr=1e-4, 20 eps, seed 42. | Val Dice: 0.3609, Sens: 0.2782, Pred/GT: 0.542. | ~2.1 GB VRAM / ~13 min | Volume underprediction (1.85×); FN dominance. |
| **EXP-1** | Aspect-Ratio Letterbox Padding | Preserving native 3:2 aspect ratio via symmetric black letterboxing prevents horizontal morphological squashing (up to 33.8%), enabling isotropic conv kernels to recognize circular lesions (MAs, dot hemorrhages). | Preprocessing: Letterbox padding to 512×512 (aspect-ratio preserved) vs squashed direct resize. | Same model (SharedUNet), same unweighted BCE+Dice, same bs=2, lr=1e-4, 20 eps, no aug, seed 42. | Val Global Dice > 0.3609; Val Macro Dice > 0.3191; Pred/GT ratio improvement. | ~2.1 GB VRAM / ~13 min | Black padding bars reduce effective retina pixel area by ~25–30%. |
| **EXP-2** | Anti-Erasure Mask Downsampling | Using area-pooling / max-pooling downsampling on raw binary masks instead of single-point `NEAREST` prevents erasing the 9.5% of MAs and 8.3% of EXs that vanish during resizing. | Mask downsampling: Area-preserving max-pooled downsampling vs standard `Image.NEAREST`. | Winner preprocessing of EXP-1, same model, loss, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice, Sensitivity on small lesions, component retention = 100%. | ~2.1 GB VRAM / ~13 min | Small single-pixel lesions may slightly expand in relative area, increasing false positive boundary pixels. |
| **EXP-3** | Positive-Weighted BCE (`pos_weight = 5.0`) | Giving positive lesion pixels 5× gradient weight in `BCEWithLogitsLoss` counteracts the 310:1 background imbalance, penalizing FNs and pulling logits out of the $p < 0.01$ suppression trap. | Loss: BCE `pos_weight = 5.0` + Dice vs unweighted BCE (`pos_weight = 1.0`) + Dice. | Best preprocessing from Track 1, same model, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice, Val Sensitivity > 0.35, Pred/GT ratio closer to 1.0. | ~2.1 GB VRAM / ~13 min | Too aggressive weighting could induce excessive false positives, dropping precision below 0.35. |
| **EXP-4** | Positive-Weighted BCE (`pos_weight = 10.0`) | Higher positive weight ($10\times$) provides stronger gradient push to detect faint, low-contrast hemorrhages and microaneurysms that remained suppressed at weight 5. | Loss: BCE `pos_weight = 10.0` + Dice. | Same as EXP-3 except `pos_weight = 10.0`. | Val Global Dice, Val Sensitivity > 0.40, Pred/GT ratio. | ~2.1 GB VRAM / ~13 min | Specificity degradation or boundary bloating. |
| **EXP-5** | Focal Loss + Soft Dice | Focal Loss dynamically suppresses easy background gradients by $(1 - p_t)^\gamma$ ($\gamma=2.0$), focusing gradient backprop specifically on ambiguous, hard-to-detect lesions without static scalar weight tuning. | Loss: Focal Loss ($\gamma=2.0, \alpha=0.25$) + Soft Dice vs BCE + Dice. | Best preprocessing from Track 1, same model, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice, Precision-Recall balance. | ~2.1 GB VRAM / ~13.5 min | Focal loss hyper-parameters ($\alpha, \gamma$) may require fine-tuning for extreme 310:1 imbalance. |
| **EXP-6** | Asymmetric Tversky Loss ($\beta = 0.7, \alpha = 0.3$) | Tversky loss directly modifies the overlap metric to weight False Negatives ($\beta=0.7$) higher than False Positives ($\alpha=0.3$), penalizing under-segmentation directly in the overlap denominator. | Loss: Tversky Loss ($\alpha=0.3, \beta=0.7$) vs standard symmetric soft Dice ($\alpha=0.5, \beta=0.5$). | Best preprocessing from Track 1, same model, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice, Val Sensitivity > 0.35, Pred/GT ratio. | ~2.1 GB VRAM / ~13 min | Under-penalizing false positives may blur lesion boundaries. |
| **EXP-7** | Ophthalmic Safe Data Augmentation | The 31M parameter U-Net overfits on 383 training images. Rotation-invariant augmentations (horizontal/vertical flips, 90° rotations, ±10% scale/brightness jitter) expand effective training diversity. | Training pipeline: Safe medical augmentation pipeline vs static images. | Best preprocessing and loss from Tracks 1 & 2, same model, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice, Gap between Train and Val loss, validation stability. | ~2.1 GB VRAM / ~14 min | Excessive geometric or color distortion could corrupt fine microaneurysm patterns. |
| **EXP-8** | Synergistic Best Combination | Combining optimal preprocessing (Track 1) + optimal loss rebalancing (Track 2) + safe augmentation (Track 3) yields compounding gains across all metrics. | Multi-factor integration: Best preprocessing + Best loss + Safe augmentation. | SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Val Global Dice $\ge 0.42$, Val Sens $\ge 0.38$, Pred/GT $\approx 0.8 - 1.1$. | ~2.1 GB VRAM / ~14 min | Unintended negative cross-factor interactions. |
| **EXP-8-Test** *(Final)* | Locked Test Set Evaluation | Verifies whether the best validation configuration generalizes to unseen test fundus images without data leakage. | Single evaluation on `DDR test split` using the winning EXP-8 checkpoint. | Fixed best checkpoint from EXP-8. Threshold fixed to 0.50 (or validation-optimal threshold). | Test Dice (target: $>0.32$ vs baseline $0.2769$), Test IoU, Test Sensitivity. | ~0.5 GB VRAM / ~1 min | Distribution shift between DDR valid and test splits. |

---

## 5. Distinction Between Measured Findings and Hypotheses

### Measured Findings (Empirically Validated from Diagnostics):
1. **Severe Area Under-prediction:** Baseline predicted lesion area is only 0.1745% of image pixels vs 0.3220% ground truth (1.85× under-prediction factor).
2. **Threshold Ineffectiveness:** Lowering threshold $\tau$ to 0.05 only raises sensitivity by +3.8% while reducing Dice and precision.
3. **Probability Bimodality:** 65.88% of GT lesion pixels and 91.27% of false negatives have predicted probability $p < 0.01$.
4. **Component Downsampling Erasure:** Nearest-neighbor resizing erases 5.18% of all lesion components and 9.47% of microaneurysms.
5. **Aspect Ratio Distortion:** Direct resizing squashes rectangular images (up to 1.51 aspect ratio) horizontally by up to 33.8%.
6. **Metric Formulation:** Global micro-pooled validation Dice is 0.3609, whereas Macro-averaged validation Dice is 0.3191.

### Hypotheses (To Be Tested Systematically in Phase 2B):
1. **Hypothesis H1 (EXP-1):** Letterbox padding to preserve native aspect ratios will reduce false negatives for compact, circular lesions (microaneurysms and hard exudates) by maintaining rotational feature symmetry.
2. **Hypothesis H2 (EXP-2):** Area-preserving mask downsampling will supply missing ground-truth supervision for the ~9.5% of small lesions currently erased, directly lifting microaneurysm recall.
3. **Hypothesis H3 (EXP-3 / EXP-4):** Positive-weighted BCE will overcome background gradient dominance, driving logits out of the $p < 0.01$ suppression trap and increasing the Pred/GT ratio toward 1.0.
4. **Hypothesis H4 (EXP-5 / EXP-6):** Asymmetric losses (Focal or Tversky) will produce smoother lesion boundary transitions and higher sensitivity than scalar BCE reweighting.
5. **Hypothesis H5 (EXP-7):** Random spatial and intensity augmentations will regularize the 31M parameter network, mitigating overfitting on the 383 training images and closing the validation-to-test performance gap.
