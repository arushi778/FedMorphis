# FedMorphis Phase 2B — EXP-1 Implementation-Readiness Audit

**Date:** 2026-10-18  
**Experiment Identifier:** `EXP-1` (Aspect-Ratio-Preserving Letterbox Preprocessing vs. Squashed Resize)  
**Location:** `results/DR/experiments/exp1_readiness_audit.md`  
**Reference Document:** `results/DR/experiments/phase2b_experiment_plan_v2.md`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB Physical GDDR6 VRAM, PyTorch 2.7.1 + CUDA 12.6)  
**Baseline Anchor:** Epoch 16 Checkpoint (`results/DR/checkpoints/best_dr_unet.pth`, Val Dice: 0.3609)  

---

## 1. Executive Summary & Audit Scope

This readiness audit evaluates whether the FedMorphis codebase and execution environment are mathematically, procedurally, and computationally prepared to implement and execute **Phase 2B EXP-1**. 

EXP-1 tests the hypothesis that replacing non-isotropic $512 \times 512$ squashed resizing with **aspect-ratio-preserving letterbox padding** eliminates horizontal lesion distortion (up to 33.8%), improving the detection of compact, circular lesions (microaneurysms and hard exudates) without confounding the comparison.

### Audit Summary:
- **Architecture & Baseline Safety:** Fully guaranteed. Zero modifications to `src/shared_unet.py` or existing baseline checkpoints.
- **Single-Variable Isolation:** Verified. Resolution ($512 \times 512$), interpolations (`BILINEAR` image, `NEAREST` mask), batch size ($2$), optimizer, and loss weights are identical to baseline.
- **Dataset Loader Gap Identified:** The active `src/dr_segmentation_dataset.py` currently returns only `(image, mask)` without valid-retinal padding masks ($M_{\text{valid}}$) or unpadding coordinates. A non-invasive extension is required before training can commence.
- **Hardware Margin:** Extremely tight. Baseline peak allocation is **2.93 GB** and reservation is **3.94 GB** ($98.5\%$ of 4.00 GB). Memory must be managed strictly.

---

## 2. Technical Audit of the Seven Readiness Criteria

### Item 1: Consistent Implementation of Aspect-Ratio Preserving Letterboxing

For any input fundus image of native dimensions $(W, H)$ and square target canvas $S = 512$:
1. **Isotropic Scaling Factor:**
   $$s = \min\left(\frac{S}{W}, \frac{S}{H}\right)$$
2. **Scaled Dimensions:**
   $$W_{\text{new}} = \text{round}(W \cdot s), \quad H_{\text{new}} = \text{round}(H \cdot s)$$
   *Constraint:* Because $s \le \frac{S}{\max(W, H)}$, $\max(W_{\text{new}}, H_{\text{new}}) = S = 512$, and $\min(W_{\text{new}}, H_{\text{new}}) \le 512$.
3. **Centered Padding Offsets:**
   $$\text{pad}_x = \left\lfloor \frac{S - W_{\text{new}}}{2} \right\rfloor, \quad \text{pad}_y = \left\lfloor \frac{S - H_{\text{new}}}{2} \right\rfloor$$
4. **RGB Fundus Image Transformation:**
   - Resized via `Image.Resampling.BILINEAR` to $(W_{\text{new}}, H_{\text{new}})$.
   - Embedded onto a $512 \times 512 \times 3$ zero-filled canvas (RGB black: $(0, 0, 0)$) at offset $(\text{pad}_x, \text{pad}_y)$.
   - Normalized to range $[0.0, 1.0]$ in tensor shape `[3, 512, 512]`.
5. **Binary Lesion Mask Transformation:**
   - Each constituent lesion mask (`EX`, `HE`, `MA`, `SE`) is resized via `Image.Resampling.NEAREST` to $(W_{\text{new}}, H_{\text{new}})$.
   - Binary union computed via `np.maximum`.
   - Embedded onto a $512 \times 512 \times 1$ zero-filled canvas (background: $0.0$) at identical offset $(\text{pad}_x, \text{pad}_y)$.
6. **Mathematical Consistency:**
   Because $s$, $W_{\text{new}}$, $H_{\text{new}}$, $\text{pad}_x$, and $\text{pad}_y$ depend solely on the native $(W, H)$ and target $S=512$, the image and all lesion channels share the **exact same affine transformation**, guaranteeing zero pixel shift or boundary desynchronization.

---

### Item 2: Valid-Retinal Indicator Mask ($M_{\text{valid}}$) in Loss and Evaluation Metrics

Letterboxing introduces black padding bars (comprising $\approx 25-33\%$ of the $512 \times 512$ canvas for $3:2$ aspect ratio images). If treated naively, these padding pixels would give the network artificial "easy negative" gradients and artificially inflate True Negatives ($TN$) and Specificity to $>0.9999$.

#### Formulation of the Valid-Retinal Mask:
$$M_{\text{valid}}(x, y) = \begin{cases} 1.0 & \text{if } \text{pad}_x \le x < \text{pad}_x + W_{\text{new}} \text{ and } \text{pad}_y \le y < \text{pad}_y + H_{\text{new}} \\ 0.0 & \text{otherwise} \end{cases}$$

#### Exclusion from Training Loss:
1. **Masked BCE Loss:**
   $$\ell_{\text{BCE}}(z, y) = \text{binary\_cross\_entropy\_with\_logits}(z, y, \text{reduction}='none')$$
   $$L_{\text{BCE, valid}} = \frac{\sum_{c, h, w} M_{\text{valid}}(h, w) \cdot \ell_{\text{BCE}}(z_{c,h,w}, y_{c,h,w})}{\sum_{h, w} M_{\text{valid}}(h, w) + \epsilon}$$
2. **Masked Soft Dice Loss:**
   Computed strictly over the valid spatial indices $K_{\text{valid}} = \{(h, w) \mid M_{\text{valid}}(h, w) = 1\}$:
   $$p_k = \sigma(z_k), \quad y_k \in \{0, 1\} \quad \forall k \in K_{\text{valid}}$$
   $$\text{Dice}_{\text{valid}} = \frac{2 \sum_{k \in K_{\text{valid}}} p_k y_k + \epsilon}{\sum_{k \in K_{\text{valid}}} p_k + \sum_{k \in K_{\text{valid}}} y_k + \epsilon}$$
   $$L_{\text{Dice, valid}} = 1.0 - \text{Dice}_{\text{valid}}$$
3. **Combined Training Loss:**
   $$L_{\text{combined, valid}} = L_{\text{BCE, valid}} + L_{\text{Dice, valid}}$$

#### Exclusion from All Seven Evaluation Metrics:
Inside `compute_metrics(logits, targets, valid_mask=None, threshold=0.5)`:
- If `valid_mask` is provided, both binary predictions $(\sigma(z) > \tau)$ and ground truth $y$ are flattened **only across valid pixels** ($M_{\text{valid}} = 1$).
- **Impact on Confusion Matrix:**
  - $\text{TP} = \sum_{k \in K_{\text{valid}}} \hat{y}_k \cdot y_k$
  - $\text{FP} = \sum_{k \in K_{\text{valid}}} \hat{y}_k \cdot (1 - y_k)$
  - $\text{FN} = \sum_{k \in K_{\text{valid}}} (1 - \hat{y}_k) \cdot y_k$
  - $\text{TN} = \sum_{k \in K_{\text{valid}}} (1 - \hat{y}_k) \cdot (1 - y_k)$
- **Result:** Padding pixels contribute **zero** counts to $\text{TP}$, $\text{FP}$, $\text{FN}$, and $\text{TN}$. Specificity, Precision, Dice, IoU, Sensitivity, and Pred/GT area ratios reflect strictly the authentic retinal field.

---

### Item 3: Dataset Loader Information Exposure Audit

We audited `src/dr_segmentation_dataset.py` (lines 165–204):
```python
def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
    return image_tensor, mask_tensor
```

#### Identified Gap:
- In the existing loader, `__getitem__` returns strictly a 2-tuple: `(image_tensor, mask_tensor)`.
- It does **not** return $M_{\text{valid}}$, nor does it return $(\text{pad}_x, \text{pad}_y, W_{\text{new}}, H_{\text{new}})$, nor native resolution $(W, H)$.
- While `get_sample_info(index)` exists, standard `DataLoader` batching cannot collate variable raw image dimensions without returning tensor masks.

#### Resolution Without Modifying Existing Code:
To strictly preserve the existing Phase 2 baseline script and dataset loader, we will implement an extended dataset class **`DDRLetterboxDataset`** in a dedicated module `src/dr_letterbox_dataset.py` (or inside the experiment directory `results/DR/experiments/exp01_letterbox_padding/dataset.py`).
- `__getitem__` will return a standardized 3-tuple:
  ```python
  return image_tensor, mask_tensor, valid_mask_tensor
  ```
- Shape: `image_tensor: [3, 512, 512]`, `mask_tensor: [1, 512, 512]`, `valid_mask_tensor: [1, 512, 512]`.
- All tensors stack into regular batches of shape `[B, C, 512, 512]`, enabling vectorized masked operations.

---

### Item 4: Single-Variable Isolation Check

To ensure scientific validity, EXP-1 must isolate **aspect-ratio preservation** as the sole independent variable:

| Parameter | Phase 2 Baseline (Control) | Phase 2B EXP-1 (Treatment) | Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `SharedUNet` (31,043,521 params) | `SharedUNet` (31,043,521 params) | **Identical** |
| **Canvas Dimensions** | $512 \times 512$ | $512 \times 512$ | **Identical** |
| **Image Interpolation** | `Image.Resampling.BILINEAR` | `Image.Resampling.BILINEAR` | **Identical** |
| **Mask Interpolation** | `Image.Resampling.NEAREST` | `Image.Resampling.NEAREST` | **Identical** |
| **Mask Union Method** | `np.maximum` across EX, HE, MA, SE | `np.maximum` across EX, HE, MA, SE | **Identical** |
| **Loss Function** | Unweighted BCE + Soft Dice (1:1) | Unweighted BCE + Soft Dice (1:1) on valid pixels | **Mathematically Equivalent** |
| **Batch Size** | 2 | 2 | **Identical** |
| **Learning Rate & Optimizer** | Adam, $\text{lr} = 10^{-4}$ | Adam, $\text{lr} = 10^{-4}$ | **Identical** |
| **Epochs** | 20 | 20 | **Identical** |
| **Random Seed** | 42 | 42 | **Identical** |
| **Data Augmentation** | None | None | **Identical** |
| **Spatial Scaling** | **Non-isotropic squashing ($s_x \ne s_y$)** | **Isotropic scaling with letterbox ($s_x = s_y = s$)** | **ISOLATED VARIABLE** |

**Conclusion:** The comparison changes strictly one variable.

---

### Item 5: Loss Function Mathematical Equivalence on Unpadded Images

Consider the proposed masked loss function:
$$L_{\text{BCE, masked}}(z, y, M) = \frac{\sum_{i} M_i \cdot \ell_{\text{BCE}}(z_i, y_i)}{\sum_i M_i + \epsilon}$$
$$L_{\text{Dice, masked}}(z, y, M) = 1.0 - \frac{2 \sum_i M_i p_i y_i + \epsilon}{\sum_i M_i p_i + \sum_i M_i y_i + \epsilon}$$

- **Proof of Backward Equivalence:**
  If an image has no padding (i.e., unpadded $512 \times 512$ baseline image), $M_i = 1.0$ for all $i \in \{1, \dots, N\}$.
  Then:
  $$\sum_i M_i = N$$
  $$L_{\text{BCE, masked}} = \frac{1}{N} \sum_{i=1}^N \ell_{\text{BCE}}(z_i, y_i) \equiv \text{nn.functional.binary\_cross\_entropy\_with\_logits}(z, y)$$
  $$L_{\text{Dice, masked}} = 1.0 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon} \equiv \text{baseline } L_{\text{Dice}}$$
- **Conclusion:** If `valid_mask is None` or all ones, the masked loss function produces **exact numerical equivalence** to the baseline loss function.

---

### Item 6: Execution Isolation & Checkpoint Preservation

To ensure the supervised baseline and vessel-segmentation files remain unaltered:
1. **Isolated Output Directory:**
   ```text
   results/DR/experiments/exp01_letterbox_padding/
   ├── checkpoints/
   │   ├── best_model.pth           # EXP-1 best checkpoint
   │   └── final_model.pth          # EXP-1 final epoch checkpoint
   ├── val_predictions/             # 6 qualitative validation panels
   ├── config.json                  # Complete recorded configuration
   ├── training_history.json        # Epoch-by-epoch loss & 7 metrics
   ├── metric_curves.png            # Visual metric trajectories
   └── exp01_vs_baseline_report.txt # Comparative evaluation summary
   ```
2. **Preservation of Baseline:**
   - Baseline checkpoint `results/DR/checkpoints/best_dr_unet.pth` is opened strictly in read-only mode (`weights_only=True`) and never targeted by any write stream.
   - Baseline test metrics file `results/DR/dr_local_test_metrics.txt` remains read-only.
3. **Vessel Segmentation Integrity:**
   - Zero files matching `*drive*`, `*chase*`, `*stare*`, or `*fives*` are imported, modified, or executed.

---

### Item 7: CUDA Peak Memory Measurement & Safe OOM Failure Protocol

#### Hardware Measurement Verification:
- Baseline verified peak allocated memory: **2.93 GB**.
- Baseline verified peak reserved memory: **3.94 GB** on a **4.00 GB physical VRAM** device.
- Available headroom: $\approx 60\text{ MB}$ reserved headroom.

#### Memory Instrumentation Protocol for EXP-1:
1. **Per-Epoch Memory Tracking:**
   At the end of each epoch, the training script will log:
   ```python
   alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
   res_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
   ```
2. **Pre-Flight Memory Dry Run (2 Batches):**
   Before initiating the 20-epoch training loop, the script will execute 2 dummy forward and backward passes using the letterboxed dataset. If peak allocated memory exceeds $3.20\text{ GB}$, training will abort immediately before epoch 1 starts.
3. **Safe Failure Handler (`try...except torch.cuda.OutOfMemoryError`):**
   ```python
   try:
       optimizer.zero_grad()
       logits = model(images)
       loss = masked_combined_loss(logits, masks, valid_masks)
       loss.backward()
       optimizer.step()
   except torch.cuda.OutOfMemoryError as e:
       print("\n[CRITICAL ERROR] CUDA Out of Memory encountered in EXP-1!")
       torch.cuda.empty_cache()
       oom_log = {
           "status": "FAILED_OOM",
           "batch_size": 2,
           "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
           "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
           "recommendation": "Activate torch.amp.autocast('cuda') for mixed precision."
       }
       with open(exp_dir / "oom_failure_report.json", "w") as f:
           json.dump(oom_log, f, indent=2)
       sys.exit(1)
   ```
4. **Automatic Memory Optimization:**
   - `torch.cuda.empty_cache()` executed after the validation pass at every epoch.
   - `num_workers = 0` to prevent Windows shared memory overhead.
   - `pin_memory = False` if host pinned pool competes with GPU driver memory.

---

## 3. Exact Proposed File Additions (Pre-Implementation Plan)

To implement EXP-1 without modifying any existing files:

1. **New Module:** `src/dr_letterbox_dataset.py`
   - Implements `DDRLetterboxDataset(DDRLesionDataset)`.
   - Overrides `__getitem__` to compute isotropic scaling factor $s$, centered offsets $(\text{pad}_x, \text{pad}_y)$, and return `(image_tensor, mask_tensor, valid_mask_tensor)`.
   - Leaves `src/dr_segmentation_dataset.py` 100% untouched.
2. **New Training Script:** `src/train_exp01_letterbox.py`
   - Reuses `SharedUNet` from `src/shared_unet.py`.
   - Uses `DDRLetterboxDataset`.
   - Implements masked BCE + masked Dice loss and masked evaluation metrics.
   - Saves all checkpoints, logs, and plots to `results/DR/experiments/exp01_letterbox_padding/`.
   - Leaves `src/train_dr_segmentation.py` 100% untouched.

---

## 4. Final Recommendation: GO / NO-GO

### **RECOMMENDATION: GO (Conditional on Isolated Script Architecture)**

**Rationale:**
1. **Scientific Validity:** EXP-1 isolates a single, physically grounded variable (aspect-ratio distortion) directly motivated by the diagnostic audit.
2. **Mathematical Rigor:** The valid-retinal mask $M_{\text{valid}}$ rigorously prevents padding pixels from corrupting both the loss gradients and evaluation metrics.
3. **Safety of Codebase:** By implementing the letterbox dataset loader and training harness as standalone files (`src/dr_letterbox_dataset.py` and `src/train_exp01_letterbox.py`), the Phase 2 baseline checkpoints, test metrics, and vessel files remain strictly unmodified.
4. **Computational Feasibility:** The memory footprint is verified at $512 \times 512$ with batch size 2, accompanied by pre-flight dry runs and safe OOM exception handling.
5. **Locked Test Split:** The test split remains completely untouched, preserving evaluation integrity.

*We are ready to implement the isolated scripts and launch EXP-1 upon your instruction.*
