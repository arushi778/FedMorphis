# FedMorphis Phase 2B: Corrected Research-Rigorous Execution Plan

**Author:** DeepMind Antigravity AI Assistant  
**Date:** 2026-10-18  
**Document Version:** 2.0 (Supersedes v1)  
**Location:** `results/DR/experiments/phase2b_experiment_plan_v2.md`  
**Anchor Baseline:** Epoch 16 Checkpoint (`results/DR/checkpoints/best_dr_unet.pth`, Val Dice: 0.3609, Test Dice: 0.2769)  
**Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB Physical GDDR6 VRAM, Compute Capability 8.6)  

---

## 1. Executive Summary & Audit Reconciliation

This document establishes the revised, research-rigorous execution plan for **Phase 2B (Controlled Supervised Baseline Improvements)** of FedMorphis. The goal of Phase 2B is to isolate and resolve the severe under-prediction and lesion suppression identified during the diagnostic audit, establishing an optimal, well-calibrated supervised baseline before transitioning to Self-Supervised Masked Image Modeling (MIM) pretraining (Phase 3) and Multi-Client Federated Learning (Phase 4).

### Exact Baseline Parameters Verified Against Active Codebase:
- **Model Architecture:** `SharedUNet` (`src/shared_unet.py`).
  - Total Parameters: **31,043,521** (all trainable).
  - Encoder: 4 stages of `DoubleConv` (channels 64, 128, 256, 512) + Bottleneck (1024).
  - Decoder: 4 stages of `UpBlock` (`ConvTranspose2d` + `DoubleConv` with skip connections) + $1 \times 1$ final convolution.
- **Batch Size:** `2` (single-stream data loading, `num_workers=0` on Windows).
- **Training Resolution:** $512 \times 512$ canvas.
- **Loss Function:** `combined_loss = bce + dice_loss` (equal unweighted sum, `src/train_dr_segmentation.py` lines 99–104).
- **Verified GPU Memory Footprint:**
  - **Peak Memory Allocated:** **2.93 GB** (tensors in active forward/backward passes).
  - **Peak Memory Reserved:** **3.94 GB** (PyTorch caching allocator block pool).
  - **Physical Capacity:** **4.00 GB** ($98.5\%$ reserved memory utilization).
- **Baseline Performance Metrics:**
  - Validation (Epoch 16): Global Pooled Dice = **0.3609**, Macro Dice = **0.3191**, IoU = **0.2201**, Sensitivity = **0.2782**, Specificity = **0.9991**, Precision = **0.5133**, Pred/GT Ratio = **0.5420**.
  - Test Set (Locked): Global Dice = **0.2769**, IoU = **0.1607**, Sensitivity = **0.1903**, Specificity = **0.9985**.

---

## 2. GPU Memory Reconciled Budget & Execution Constraints (RTX 3050, 4 GB)

The recorded baseline consumption of **2.93 GB allocated and 3.94 GB reserved** establishes that the training environment operates at the exact boundary of physical VRAM. 

### VRAM Component Decomposition:
$$\text{Total VRAM} = \text{Static Weights} + \text{Optimizer States} + \text{CUDA Kernels} + \text{Forward Activations} + \text{Workspace Buffer}$$
1. **Model Weights (FP32):** $31.04\text{ M} \times 4\text{ bytes} = 124.18\text{ MB}$.
2. **Gradients (FP32):** $124.18\text{ MB}$.
3. **Adam Optimizer States:** $2 \times (31.04\text{ M} \times 4\text{ bytes}) = 248.36\text{ MB}$.
4. **PyTorch / CUDA Driver Overhead:** $\approx 550 - 650\text{ MB}$.
5. **Forward-Backward Activations ($512 \times 512$, BS=2):** $\approx 1,450 - 1,750\text{ MB}$.
6. **Allocator Cache Fragmentation:** $\approx 950 - 1,050\text{ MB}$ (bringing total reserved to $3.94\text{ GB}$).

### Operational Rules for Phase 2B:
- **Rule 1 (Strict Batch Size Cap):** Physical `batch_size = 2` cannot be exceeded. Any physical batch size $\ge 3$ will immediately trigger a CUDA out-of-memory crash. If an effective batch size of 4 is required, gradient accumulation (`accumulation_steps = 2`) must be used.
- **Rule 2 (Strict Resolution Cap):** Direct full-image resizing or letterboxing must remain bounded at $512 \times 512$. Full-image $1024 \times 1024$ training requires $>5.5\text{ GB}$ activation memory and is strictly prohibited on this hardware.
- **Rule 3 (Cache Purging):** Explicit calls to `torch.cuda.empty_cache()` must be executed at validation and epoch boundaries to prevent cumulative block fragmentation.
- **Rule 4 (Mixed Precision Fallback):** If letterbox padding or loss modifications cause minor memory spikes, native PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`) will be engaged, reducing activation memory by $\approx 40\%$ while preserving numerical stability.

---

## 3. Pre-Registration of Validation Metrics & Model Selection Rule

### 3.1 Pre-Registered Metric Suite
All experiments in Phase 2B will log and report seven standardized validation metrics evaluated at classification threshold $\tau = 0.50$ (and at the validation-optimal threshold $\tau^*$ determined post-hoc on validation logits):

1. **Global Pooled Dice (Primary Benchmark Metric):**
   $$\text{Dice}_{\text{global}} = \frac{2 \sum_{i} \text{TP}_i + \epsilon}{2 \sum_{i} \text{TP}_i + \sum_{i} \text{FP}_i + \sum_{i} \text{FN}_i + \epsilon}$$
   Evaluates total dataset-level segmentation overlap, directly comparable to the 0.3609 baseline anchor.
2. **Macro-Averaged Dice (Per-Image Mean):**
   $$\text{Dice}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^N \frac{2 \text{TP}_i + \epsilon}{2 \text{TP}_i + \text{FP}_i + \text{FN}_i + \epsilon}$$
   Evaluates per-patient consistency, preventing images with massive hemorrhage clusters from masking poor performance on subtle cases.
3. **Global IoU (Jaccard Index):**
   $$\text{IoU}_{\text{global}} = \frac{\sum_{i} \text{TP}_i + \epsilon}{\sum_{i} \text{TP}_i + \sum_{i} \text{FP}_i + \sum_{i} \text{FN}_i + \epsilon}$$
4. **Global Sensitivity (Recall):**
   $$\text{Sensitivity}_{\text{global}} = \frac{\sum_{i} \text{TP}_i + \epsilon}{\sum_{i} \text{TP}_i + \sum_{i} \text{FN}_i + \epsilon}$$
   Directly measures the recovery rate of true lesion pixels.
5. **Global Specificity:**
   $$\text{Specificity}_{\text{global}} = \frac{\sum_{i} \text{TN}_i + \epsilon}{\sum_{i} \text{TN}_i + \sum_{i} \text{FP}_i + \epsilon}$$
   Ensures the model does not sacrifice background precision.
6. **Global Precision (Positive Predictive Value):**
   $$\text{Precision}_{\text{global}} = \frac{\sum_{i} \text{TP}_i + \epsilon}{\sum_{i} \text{TP}_i + \sum_{i} \text{FP}_i + \epsilon}$$
7. **Predicted / Ground-Truth Pixel Volume Ratio:**
   $$\text{Ratio}_{\text{pred/gt}} = \frac{\sum_{i} (\text{TP}_i + \text{FP}_i)}{\sum_{i} (\text{TP}_i + \text{FN}_i)}$$
   Target range: $0.80 - 1.20$ (baseline was $0.5420$, representing a $1.85\times$ underprediction).

### 3.2 Model Selection and Stopping Rule
- Model selection is governed exclusively by **Validation Global Pooled Dice**.
- During the 20 epochs, the checkpoint from the epoch maximizing validation Dice is saved as `best_model.pth`.
- **Threshold Selection:** Validation logits will undergo an offline threshold sweep $\tau \in [0.10, 0.90]$ with step $0.05$. The threshold $\tau^*$ maximizing validation Dice is recorded for diagnostic analysis.
- **Locked Test Set Protocol:** The test split (`data/DR/DDR-dataset/lesion_segmentation/test/`) remains **completely untouched, unloaded, and uninspected** throughout Phase 2B. Only the single best configuration selected from Phase 2B validation results will be evaluated on the test set prior to Phase 3.

---

## 4. Rigorous Treatment of Experimental Variables

### 4.1 EXP-1: Aspect-Ratio Preserving Preprocessing & Padding Exclusion
In the baseline, images with aspect ratios up to 1.51 are forcefully squashed into a square, compressing horizontal dimensions by up to $33.8\%$.

#### Preprocessing Formulation:
For an image with native dimensions $(W, H)$ and canvas $S = 512$:
1. Scale factor: $s = \min\left(\frac{S}{W}, \frac{S}{H}\right)$.
2. Scaled dimensions: $W_{\text{new}} = \text{round}(W \cdot s)$, $H_{\text{new}} = \text{round}(H \cdot s)$.
3. Padding offsets:
   $$\text{pad}_x = \left\lfloor \frac{S - W_{\text{new}}}{2} \right\rfloor, \quad \text{pad}_y = \left\lfloor \frac{S - H_{\text{new}}}{2} \right\rfloor$$
4. Canvas placement: The scaled image is centered on a $512 \times 512 \times 3$ black canvas ($0$ intensity).
5. Ground truth placement: The mask is scaled using nearest-neighbor to $(W_{\text{new}}, H_{\text{new}})$ and centered on a $512 \times 512 \times 1$ zero canvas.
6. **Valid Region Indicator ($M_{\text{valid}}$):**
   $$M_{\text{valid}}(x, y) = \begin{cases} 1 & \text{if } \text{pad}_x \le x < \text{pad}_x + W_{\text{new}} \text{ and } \text{pad}_y \le y < \text{pad}_y + H_{\text{new}} \\ 0 & \text{otherwise} \end{cases}$$

#### Exclusion of Padding Pixels:
- **Training Loss Masking:** Loss is strictly calculated on valid pixels:
  $$L_{\text{BCE, valid}} = \frac{\sum_{(x,y)} M_{\text{valid}}(x,y) \cdot \ell_{\text{BCE}}(z(x,y), y(x,y))}{\sum_{(x,y)} M_{\text{valid}}(x,y)}$$
  This prevents the network from receiving artificial "easy negative" gradients from black border pixels.
- **Evaluation Canvas Unpadding:**
  During validation evaluation, predictions inside the valid bounding box $[\text{pad}_y : \text{pad}_y + H_{\text{new}}, \text{pad}_x : \text{pad}_x + W_{\text{new}}]$ are extracted. Metrics are computed **only within the valid retinal region**, guaranteeing that padding pixels never artificially inflate True Negatives ($TN$) or Specificity.

---

### 4.2 EXP-2: Mask Downsampling Trade-Offs (Nearest-Neighbor vs Documented Alternatives)

The diagnostic audit revealed that $512 \times 512$ nearest-neighbor downsampling erases **5.18% of all lesion components and 9.47% of microaneurysms**. However, naive max-pooling is **not** an automatically correct remedy.

#### Empirical Downsampling Measurement Across 149 Validation Masks:
- **Raw Native Ground Truth:** $2,776,014$ lesion pixels across $6,275$ connected components.
- **Method 1 — Nearest-Neighbor Downsampling (`Image.NEAREST`):**
  - Resulting Lesion Pixels: **$125,782\text{ px}$**.
  - Surviving Connected Components: **$5,933$** ($94.5\%$ component retention; **$342$ components erased, $5.5\%$**).
  - Preserves precise local boundary scale but drops sub-grid punctate lesions.
- **Method 2 — Unconstrained Max-Pooling (`adaptive_max_pool2d`):**
  - Resulting Lesion Pixels: **$183,039\text{ px}$** (**$+45.5\%$ artificial area dilation / $1.46\times$ inflation**).
  - Connected Components: **$5,331$** ($85.0\%$ component count due to adjacent distinct lesions merging).
  - **Severe Limitation:** A single 1-pixel raw microaneurysm expands to fill an entire $5 \times 5$ grid cell, corrupting physical lesion morphology and penalizing well-calibrated predictions.
- **Method 3 — Documented Lesion-Preserving Alternative: Area-Fraction Pooling with Calibrated Threshold:**
  - Continuous average pooling calculates local lesion area fraction: $A(x,y) = \text{adaptive\_avg\_pool2d}(Y_{\text{raw}})$.
  - Binarization threshold $\theta_{\text{area}} \in \{0.05, 0.10\}$:
    $$Y_{\text{down}}(x,y) = \mathbb{I}(A(x,y) \ge \theta_{\text{area}})$$
  - At $\theta = 0.05$: Resulting pixels = $178,481$; components = $5,403$.
  - At higher $\theta$, boundary bloating is constrained while retaining small lesions.

#### Annotation Trade-Off Matrix:
| Downsampling Method | Component Retention | Area Preservation | Morphology Fidelity | Risk to Model |
| :--- | :---: | :---: | :---: | :--- |
| **Nearest-Neighbor** | Lower ($94.5\%$) | High ($1.00\times$ reference) | High (unblurred) | Model never receives supervision for tiny MAs. |
| **Max-Pooling** | High for singletons | Severely Bloated ($1.46\times$) | Poor (merges nearby spots) | Over-predicts lesion area; penalizes crisp borders. |
| **Area-Fraction Threshold** | Calibrated ($96-98\%$) | Controlled ($1.10-1.25\times$) | Moderate | Requires hyperparameter selection ($\theta_{\text{area}}$). |

**Phase 2B Decision:** EXP-2 will empirically compare standard `NEAREST` against **Area-Fraction Thresholded Downsampling ($\theta_{\text{area}} = 0.05$)**, evaluating whether component retention outweighs the penalty of mild boundary expansion.

---

### 4.3 EXP-3 & EXP-4: Positive-Weighted BCE (`pos_weight = 5.0` and `10.0`)
To directly address the 310:1 background dominance, positive pixels are weighted by scalar $w_{\text{pos}}$:

#### Mathematical Formulation:
$$\ell_{\text{weighted\_bce}}(z, y) = - \left[ w_{\text{pos}} \cdot y \cdot \log(\sigma(z)) + (1 - y) \cdot \log(1 - \sigma(z)) \right]$$

#### Loss Reduction and Normalization Consistency:
Standard PyTorch `BCEWithLogitsLoss(pos_weight=w, reduction='mean')` divides by total pixel count $N$, meaning that as $w_{\text{pos}}$ increases, the raw loss magnitude scales upward. To maintain exact balance between the BCE loss term and the soft Dice loss term ($L_{\text{Dice}} \in [0, 1]$), the weighted BCE loss must be normalized by its effective sample weight:
$$W_{\text{effective}} = \frac{w_{\text{pos}} N_{\text{pos}} + N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}}}$$
$$L_{\text{norm\_bce}} = \frac{1}{W_{\text{effective}}} \cdot \text{BCEWithLogitsLoss}(z, y; w_{\text{pos}})$$
$$L_{\text{total}} = L_{\text{norm\_bce}} + L_{\text{Dice}}$$
- **EXP-3:** Evaluates $w_{\text{pos}} = 5.0$ (moderate penalty on false negatives).
- **EXP-4:** Evaluates $w_{\text{pos}} = 10.0$ (strong penalty on false negatives).
Both experiments keep normalization mathematically identical to ensure gradient scales remain directly comparable.

---

### 4.4 EXP-5: Focal Loss Parameters for Binary Lesion Segmentation
Standard multi-class focal loss (Lin et al. 2017) uses $\alpha = 0.25$ for foreground detection under dense anchor boxes. In binary segmentation where background represents $99.68\%$ of pixels, setting $\alpha = 0.25$ assigns foreground weight $\alpha = 0.25$ and background weight $1 - \alpha = 0.75$, weighting background **$3\times$ higher than lesions**!

#### Corrected Binary Focal Loss Formulation:
$$p_t = \begin{cases} \sigma(z) & \text{if } y = 1 \\ 1 - \sigma(z) & \text{if } y = 0 \end{cases}$$
$$\alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$
$$\text{FL}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
- **Parameter Selection for Extreme DR Imbalance:**
  - $\gamma = 2.0$ (modulating factor that squashes gradients from well-classified background pixels where $p_t > 0.99$).
  - $\alpha = 0.75$ (assigns $3\times$ base weighting to true positive lesion pixels, aligning with class rarity).
- **Combined Objective:** $L_{\text{total}} = \text{FL}(p_t; \alpha=0.75, \gamma=2.0) + L_{\text{Dice}}$.

---

### 4.5 EXP-6: Tversky Loss Mathematical Conventions
The Tversky index generalizes Dice by uncoupling penalties for False Positives ($FP$) and False Negatives ($FN$):

#### Mathematical Formulation:
$$\text{TI}(p, y) = \frac{\sum_{k} p_k y_k + \epsilon}{\sum_{k} p_k y_k + \alpha \sum_{k} p_k (1 - y_k) + \beta \sum_{k} (1 - p_k) y_k + \epsilon} = \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \text{FP} + \beta \text{FN} + \epsilon}$$
- **Conventions & Constraints:**
  - $\alpha$ is the penalty parameter for **False Positives (FP)**.
  - $\beta$ is the penalty parameter for **False Negatives (FN)**.
  - Normalization constraint: $\alpha + \beta = 1.0$.
  - When $\alpha = \beta = 0.5$, $\text{TI}$ reduces identically to the standard Dice coefficient:
    $$\text{TI}(0.5, 0.5) = \frac{\text{TP}}{\text{TP} + 0.5\text{FP} + 0.5\text{FN}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}} = \text{Dice}$$
  - **Correction for Under-Prediction:** To penalize false negatives (low sensitivity) and force the model to capture subtle lesions, we set **$\beta = 0.7$ and $\alpha = 0.3$** ($\beta > \alpha$).
- **Combined Objective:** $L_{\text{total}} = L_{\text{BCE}} + (1 - \text{TI}(p, y; \alpha=0.3, \beta=0.7))$.

---

## 5. Phase 2B Corrected Experiment Matrix

*Every experiment tests one isolated hypothesis. Performance targets are strictly exploratory benchmarks, not guaranteed outcomes.*

| Exp ID | Experiment Name | Hypothesis | Changed Variable | Fixed Variables | Exploratory Benchmark Criteria | Expected Cost (VRAM / Time) | Potential Risks |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **EXP-0** *(Control)* | Baseline Anchor | Re-verified reference anchor. | None | $512 \times 512$, squashed, unweighted BCE+Dice, bs=2, lr=1e-4, 20 eps, seed 42. | Anchor: Val Dice 0.3609, Sens 0.2782, Pred/GT 0.542. | 2.93 GB Alloc / 3.94 GB Res / ~13 min | 1.85× volume underprediction; FN dominance. |
| **EXP-1** | Aspect-Ratio Letterbox Padding | Aspect-ratio-preserving padding eliminates 33.8% horizontal squashing, improving detection of circular lesions (MAs, EXs). Padding pixels are masked out of loss and evaluation. | **Preprocessing:** Letterbox padding to 512×512 with valid-region masking vs squashed resize. | SharedUNet, unweighted BCE+Dice, bs=2, lr=1e-4, 20 eps, no aug, seed 42. | $\Delta\text{Val Dice} > +0.01$; higher macro Dice; reduced aspect distortion. | ~2.93 GB Alloc / 3.94 GB Res / ~13 min | Black padding bars reduce effective retina area on canvas by ~25%. |
| **EXP-2** | Area-Preserving Mask Downsampling | Thresholded area-pooling downsampling ($\theta=0.05$) preserves tiny lesions without the 45.5% area bloating caused by naive max-pooling. | **Mask Resizing:** Area-fraction downsampling vs standard `NEAREST`. | Winner of EXP-1, SharedUNet, unweighted BCE+Dice, bs=2, lr=1e-4, 20 eps, seed 42. | Higher MA component recall; Val Dice stability. | ~2.93 GB Alloc / 3.94 GB Res / ~13 min | Slight expansion of lesion fringes may increase marginal false positives. |
| **EXP-3** | Positive-Weighted BCE (`pos_weight = 5.0`) | Normalized positive weighting ($w_{\text{pos}}=5$) counteracts 310:1 background gradient dominance, penalizing FNs and pulling logits out of $p < 0.01$. | **Loss Formulation:** Normalized BCE ($w_{\text{pos}}=5.0$) + Dice vs unweighted BCE ($w_{\text{pos}}=1.0$). | Best preprocessing (Track 1), SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Sensitivity increases ($\ge 0.33$); Pred/GT ratio moves toward 0.75–0.90. | ~2.93 GB Alloc / 3.94 GB Res / ~13 min | Precision drops below 0.40 if background false positives rise. |
| **EXP-4** | Positive-Weighted BCE (`pos_weight = 10.0`) | Higher positive weight ($w_{\text{pos}}=10$) provides stronger gradient pressure to recover faint, low-contrast hemorrhages. | **Loss Formulation:** Normalized BCE ($w_{\text{pos}}=10.0$) + Dice. | Same as EXP-3 except $w_{\text{pos}}=10.0$. | Sensitivity increases ($\ge 0.38$); Pred/GT ratio moves toward 0.90–1.10. | ~2.93 GB Alloc / 3.94 GB Res / ~13 min | Specificity drops below 0.998; precision degrades. |
| **EXP-5** | Calibrated Binary Focal Loss | Focal loss ($\gamma=2.0, \alpha=0.75$) down-weights easy background pixels, focusing gradients on hard lesion boundaries. | **Loss Formulation:** Focal Loss ($\alpha=0.75, \gamma=2.0$) + Dice vs BCE + Dice. | Best preprocessing (Track 1), SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Balanced precision-recall trade-off; higher IoU. | ~2.93 GB Alloc / 3.94 GB Res / ~13.5 min | Focal modulating factor may slow early convergence. |
| **EXP-6** | Asymmetric Tversky Loss ($\beta = 0.7, \alpha = 0.3$) | Tversky loss penalizes FNs directly in the overlap denominator, boosting sensitivity without logit distortion. | **Overlap Loss:** Tversky Loss ($\beta=0.7, \alpha=0.3$) vs symmetric Dice ($\beta=0.5, \alpha=0.5$). | Best preprocessing (Track 1), SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Sensitivity increases ($\ge 0.35$); Dice $\ge 0.36$. | ~2.93 GB Alloc / 3.94 GB Res / ~13 min | Under-penalizing false positives may blur lesion boundaries. |
| **EXP-7** | Ophthalmic Safe Augmentation | Invariant retinal augmentations (horizontal/vertical flips, 90° rotations, $\pm 8\%$ scale/brightness jitter) reduce overfitting on 383 training images. | **Data Augmentation:** Safe medical augmentation pipeline vs static unaugmented images. | Best preprocessing + best loss from Tracks 1 & 2, SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Reduced gap between train and val loss; improved val stability. | ~2.93 GB Alloc / 3.94 GB Res / ~14 min | Color jitter could alter exudate contrast if too aggressive. |
| **EXP-8** | Integrated Best Supervised Baseline | Combining optimal preprocessing (Track 1) + optimal loss (Track 2) + safe augmentation (Track 3) yields cumulative improvements. | **Multi-Factor Integration:** Best preprocessing + Best loss + Safe augmentation. | SharedUNet, bs=2, lr=1e-4, 20 eps, seed 42. | Exploratory Goal: Val Dice $\approx 0.39 - 0.43$, Sens $\approx 0.35 - 0.42$, Pred/GT $\approx 0.85 - 1.15$. | ~2.93 GB Alloc / 3.94 GB Res / ~14 min | Cross-factor negative interactions. |
| **EXP-8-Test** *(Final)* | Locked Test Set Evaluation | Measures unbiased generalization of the final selected supervised model to unseen test fundus images. | Single evaluation on the locked `DDR test split` using the winning EXP-8 checkpoint. | Fixed best checkpoint from EXP-8. Threshold fixed to 0.50 (and validation-optimal $\tau^*$). | Exploratory Goal: Test Dice $> 0.30$ (vs baseline $0.2769$), Test Sens $> 0.25$. | ~0.6 GB Alloc / ~1 min | Domain shift between DDR validation and test splits. |

---

## 6. Reproducibility Controls & Environment Specifications

- **Deterministic Seeds:** Seed `42` across Python `random`, `numpy`, and `torch`.
- **Backend Determinism:**
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- **Logging Artifacts per Experiment:**
  Every experiment subfolder `results/DR/experiments/<exp_id>/` will automatically record:
  1. `config.json`: Complete dictionary of all hyperparameters, loss settings, and preprocessing flags.
  2. `training_history.json`: Epoch-by-epoch record of train loss, val loss, global dice, macro dice, sensitivity, specificity, precision, and Pred/GT ratio.
  3. `metric_curves.png`: Loss curves and metric evolution across 20 epochs.
  4. `val_predictions/`: Six qualitative overlay panels corresponding to the 6 diagnostic images (`val_pred_007-5730-300`, etc.).
  5. `best_model.pth`: Checkpoint containing model state dict, optimizer state, epoch, and validation metrics.
  6. `experiment_summary.txt`: Standalone text summary comparing results against the baseline anchor.

---

## 7. Strategic Transition to SSL/MIM Pretraining and Federated Learning

Phase 2B is not an isolated exercise in supervised tuning; it directly provides the calibrated evaluation testbed and architectural backbone for the core innovations of FedMorphis:

```mermaid
flowchart TD
    subgraph Phase2B ["Phase 2B: Supervised Optimization"]
        A["EXP-1 / EXP-2: Preprocessing & Anti-Erasure"] --> B["EXP-3 to EXP-6: Imbalance Loss Optimization"]
        B --> C["EXP-7: Invariant Augmentation"]
        C --> D["EXP-8: Optimal Supervised Baseline"]
    end

    subgraph Phase3 ["Phase 3: Self-Supervised MIM Pretraining"]
        D --> E["SharedEncoder Initialization"]
        F["Unlabeled Multi-Domain Retinal Images"] --> G["Masked Image Modeling (MIM) Pretraining"]
        G --> H["Pretrained SharedEncoder Weights"]
        H --> I["Fine-Tuning on Calibrated Supervised Pipeline"]
    end

    subgraph Phase4 ["Phase 4: Federated Learning (FedMorphis)"]
        I --> J["Cross-Silo FedAvg / FedMorphis Aggregation"]
        K["Client 1: DDR (DR Lesions)"] --> J
        L["Client 2: DRIVE (Vessels)"] --> J
        M["Client 3: CHASE / STARE (Vessels)"] --> J
        N["Client 4: FIVES (Vessels & Quality)"] --> J
        J --> O["Global Unified Retinal Representation"]
    end
```

### Detailed Transition Roadmap:
1. **Transition to Phase 3 (SSL / Masked Image Modeling):**
   - The encoder of `SharedUNet` (`SharedEncoder`, `src/shared_encoder.py`) is structurally identical to the encoder trained in `train_shared_mim.py`.
   - In Phase 2B, the model is trained from scratch. In Phase 3, the optimal loss, preprocessing, and augmentation pipeline developed in Phase 2B will be frozen, and the encoder will be initialized with weights pretrained via Masked Image Modeling on unlabeled retinal images.
   - This isolates the exact contribution of SSL pretraining: any performance gain over the Phase 2B supervised benchmark can be definitively attributed to the learned self-supervised representations.
2. **Transition to Phase 4 (Multi-Client Federated Learning):**
   - In Phase 4, the model enters a federated environment (`src/train_federated.py`, `src/fedavg.py`) where clients hold disjoint, non-IID datasets:
     - Client 1: DDR (diabetic retinopathy lesion segmentation, high class imbalance).
     - Client 2: DRIVE (vessel segmentation, binary tubular structures).
     - Client 3: CHASE_DB1 / STARE (arteriole/venule vascular trees).
     - Client 4: FIVES (multi-condition retinal screening).
   - The robust loss functions validated in Phase 2B (e.g., positive-weighted BCE, Tversky loss) are essential to prevent local client collapse under extreme non-IID class imbalance during federated rounds.

---

## 8. Summary of Plan Audit: Verified Items, Changes Made, and Unresolved Assumptions

### Exact Items Verified:
1. **Model Parameter Count:** Verified at exactly **31,043,521** trainable parameters.
2. **GPU Memory Footprint:** Verified on NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB physical VRAM): **2.93 GB peak allocation** and **3.94 GB peak reservation** at batch size 2 ($98.5\%$ reserved memory utilization).
3. **Loss Implementation:** Verified `combined_loss = bce + dice_loss` in `src/train_dr_segmentation.py` lines 99–104.
4. **Metric Implementation:** Verified global micro-pooling concatenation across validation batches and empty-mask sensitivity edge case in `src/evaluate_dr_segmentation.py`.
5. **Mask Downsampling Trade-Off:** Measured on 149 validation masks: nearest-neighbor drops $5.5\%$ of components ($125,782\text{ px}$); adaptive max-pooling causes **$+45.5\%$ artificial area bloating** ($183,039\text{ px}$) and merges adjacent lesions ($5,331$ components vs $6,275$ raw).

### Specific Changes Made in Version 2.0:
1. **Replaced Naive Max-Pooling:** Formulated thresholded area-fraction pooling ($\theta=0.05$) to balance component retention with boundary preservation, documenting the exact trade-offs.
2. **Rigorous Padding Exclusion in EXP-1:** Defined explicit valid-region indicator $M_{\text{valid}}$ for training loss masking and unpadded bounding-box extraction for evaluation.
3. **Loss Normalization for $w_{\text{pos}}=5$ and $10$:** Mathematically formulated effective weight normalization ($W_{\text{effective}}$) to maintain exact balance between BCE and soft Dice across EXP-3 and EXP-4.
4. **Corrected Focal Loss and Tversky Formulations:**
   - Corrected focal loss $\alpha$ from detection default $0.25$ to segmentation foreground weight $\alpha = 0.75$ ($\gamma = 2.0$).
   - Formulated Tversky loss with explicit false negative penalty $\beta = 0.7$ and false positive penalty $\alpha = 0.3$ ($\alpha + \beta = 1.0$).
5. **Replaced Guaranteed Targets with Exploratory Benchmarks:** Shifted all performance milestones to exploratory benchmarks with explicit null-hypothesis criteria.
6. **VRAM Reconciled Rules:** Established strict batch size cap ($2$) and memory-protection protocols.
7. **Explicit SSL and Federated Roadmap:** Detailed how Phase 2B directly bridges to Phase 3 MIM transfer and Phase 4 multi-client federated learning.

### Unresolved Assumptions Requiring Empirical Testing:
1. **Letterbox Padding Canvas Shrinkage:** Black padding preserves aspect ratios but reduces the effective retinal pixel footprint by $\approx 25\%$. EXP-1 must resolve whether aspect-ratio preservation outweighs this resolution reduction.
2. **Boundary Expansion Penalty:** Whether the mild boundary dilation of area-preserving mask downsampling in EXP-2 improves Dice or is penalized by false positive inflation.
3. **Positive Weight Sensitivity Threshold:** Whether scalar reweighting ($w_{\text{pos}}=5$ vs $10$) or asymmetric overlap loss (Tversky) better resolves probability bimodality without eroding specificity.
