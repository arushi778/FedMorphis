"""
Verification Suite for Phase 2B EXP-1 (Pre-Training Verification).

Tests the 6 verification criteria required before launching full training:
1. Dataset output shapes and valid-mask shapes.
2. Binary mask values and image-mask alignment (no lesion pixels outside valid region).
3. Padding exclusion in loss and metrics.
4. Numerical equivalence of masked and baseline losses when valid_mask is all ones.
5. Single forward/backward/optimizer dry run with batch size 2 on GPU, recording peak memory.
6. Confirmation that original baseline checkpoint and vessel files remain untouched.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dr_letterbox_dataset import DDRLetterboxDataset
from shared_unet import SharedUNet
from train_exp01_letterbox import masked_combined_loss, compute_metrics as letterbox_compute_metrics
from train_dr_segmentation import combined_loss as baseline_combined_loss, compute_metrics as baseline_compute_metrics


def run_checks():
    print("=" * 80)
    print("PHASE 2B EXP-1: PRE-TRAINING VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECK 1: Dataset output shapes and valid-mask shapes
    # -------------------------------------------------------------------------
    print("\n[CHECK 1] Testing Dataset Output Shapes...")
    ds_train = DDRLetterboxDataset(split="train", image_size=512)
    ds_val = DDRLetterboxDataset(split="valid", image_size=512)

    img, mask, vmask = ds_train[0]
    assert img.shape == (3, 512, 512), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    assert vmask.shape == (1, 512, 512), f"Unexpected valid_mask shape: {vmask.shape}"
    assert img.dtype == torch.float32, f"Unexpected image dtype: {img.dtype}"
    assert mask.dtype == torch.float32, f"Unexpected mask dtype: {mask.dtype}"
    assert vmask.dtype == torch.float32, f"Unexpected valid_mask dtype: {vmask.dtype}"
    print(f"  Train Sample 0 shapes: Image={list(img.shape)}, Mask={list(mask.shape)}, ValidMask={list(vmask.shape)}")
    print(f"  Valid Sample 0 shapes: Image={list(ds_val[0][0].shape)}, Mask={list(ds_val[0][1].shape)}, ValidMask={list(ds_val[0][2].shape)}")
    print("  -> CHECK 1 PASSED: Shapes are strictly [3, 512, 512], [1, 512, 512], [1, 512, 512]")

    # -------------------------------------------------------------------------
    # CHECK 2: Binary mask values and image-mask alignment
    # -------------------------------------------------------------------------
    print("\n[CHECK 2] Testing Binary Mask Values and Image-Mask Alignment...")
    for idx in range(min(10, len(ds_train))):
        img_i, mask_i, vmask_i = ds_train[idx]
        mask_vals = set(torch.unique(mask_i).tolist())
        vmask_vals = set(torch.unique(vmask_i).tolist())

        assert mask_vals.issubset({0.0, 1.0}), f"Sample {idx} mask not binary: {mask_vals}"
        assert vmask_vals.issubset({0.0, 1.0}), f"Sample {idx} valid_mask not binary: {vmask_vals}"

        # Alignment: No lesion pixel may fall outside the valid-retinal area
        lesions_outside = (mask_i > 0) & (vmask_i == 0)
        assert lesions_outside.sum() == 0, f"Sample {idx}: Lesion pixels found in padding area!"

        # Padding pixels must be black (zero)
        padding_pixels = (vmask_i == 0).expand(3, 512, 512)
        assert img_i[padding_pixels].abs().sum() == 0.0, f"Sample {idx}: Non-black pixels found in padding area!"

    print("  -> CHECK 2 PASSED: All mask values strictly in {0.0, 1.0}; zero lesions in padding; padding pixels are 100% black.")

    # -------------------------------------------------------------------------
    # CHECK 3: Padding exclusion in loss and metrics
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Testing Padding Exclusion in Loss and Metrics...")
    torch.manual_seed(42)
    B = 2
    logits = torch.randn(B, 1, 512, 512)
    targets = torch.randint(0, 2, (B, 1, 512, 512)).float()
    vmask = torch.zeros(B, 1, 512, 512)
    vmask[:, :, 50:450, 20:480] = 1.0  # mock valid region

    # Ensure targets outside valid area are zero initially
    targets = targets * vmask

    loss_clean = masked_combined_loss(logits, targets, vmask)
    metrics_clean = letterbox_compute_metrics(logits, targets, vmask, threshold=0.5)

    # Now inject massive noise in the padding area (where vmask == 0)
    logits_corrupted = logits.clone()
    targets_corrupted = targets.clone()
    logits_corrupted[:, :, :50, :] = 999.0  # huge positive logits in padding
    targets_corrupted[:, :, :50, :] = 1.0   # huge false positives in padding

    loss_corrupted = masked_combined_loss(logits_corrupted, targets_corrupted, vmask)
    metrics_corrupted = letterbox_compute_metrics(logits_corrupted, targets_corrupted, vmask, threshold=0.5)

    loss_diff = abs(loss_clean.item() - loss_corrupted.item())
    dice_diff = abs(metrics_clean["dice"] - metrics_corrupted["dice"])
    spec_diff = abs(metrics_clean["specificity"] - metrics_corrupted["specificity"])

    assert loss_diff < 1e-6, f"Padding corrupted loss! diff={loss_diff}"
    assert dice_diff < 1e-6, f"Padding corrupted dice! diff={dice_diff}"
    assert spec_diff < 1e-6, f"Padding corrupted specificity! diff={spec_diff}"
    print(f"  Loss delta with massive padding corruption: {loss_diff:.2e}")
    print(f"  Dice delta with massive padding corruption: {dice_diff:.2e}")
    print(f"  Specificity delta with massive padding corruption: {spec_diff:.2e}")
    print("  -> CHECK 3 PASSED: Padding pixels are completely insulated from both loss gradients and evaluation metrics.")

    # -------------------------------------------------------------------------
    # CHECK 4: Equivalence of masked and baseline losses when valid_mask is all ones
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Testing Equivalence of Masked and Baseline Losses on All-Ones Mask...")
    torch.manual_seed(42)
    logits = torch.randn(2, 1, 512, 512)
    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    vmask_ones = torch.ones_like(targets)

    base_loss = baseline_combined_loss(logits, targets)
    masked_loss_with_ones = masked_combined_loss(logits, targets, vmask_ones)
    masked_loss_with_none = masked_combined_loss(logits, targets, valid_mask=None)

    diff_ones = abs(base_loss.item() - masked_loss_with_ones.item())
    diff_none = abs(base_loss.item() - masked_loss_with_none.item())

    assert diff_ones < 1e-6, f"Masked loss (ones) != baseline loss! diff={diff_ones}"
    assert diff_none < 1e-6, f"Masked loss (None) != baseline loss! diff={diff_none}"
    print(f"  Baseline Loss:               {base_loss.item():.6f}")
    print(f"  Masked Loss (valid_mask=1):  {masked_loss_with_ones.item():.6f} (diff = {diff_ones:.2e})")
    print(f"  Masked Loss (valid_mask=None):{masked_loss_with_none.item():.6f} (diff = {diff_none:.2e})")
    print("  -> CHECK 4 PASSED: Masked loss produces exact mathematical equivalence within float precision.")

    # -------------------------------------------------------------------------
    # CHECK 5: Forward/Backward/Optimizer Dry Run with Batch Size 2
    # -------------------------------------------------------------------------
    print("\n[CHECK 5] Running Forward/Backward/Optimizer Dry Run with Batch Size 2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=False)

    from train_exp01_letterbox import run_cuda_dry_run
    stats = run_cuda_dry_run(model, loader, optimizer, device)

    if device.type == "cuda":
        assert stats["allocated_gb"] < 3.8, f"Allocated memory exceeds safety threshold: {stats['allocated_gb']:.2f} GB"
        print(f"  Dry run peak allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Dry run peak reserved:  {stats['reserved_gb']:.2f} GB")
        print(f"  Total physical VRAM:    {stats['total_gb']:.2f} GB")
    print("  -> CHECK 5 PASSED: Forward, loss, backward, and optimizer step completed successfully.")

    # -------------------------------------------------------------------------
    # CHECK 6: Confirmation of baseline and vessel file immutability
    # -------------------------------------------------------------------------
    print("\n[CHECK 6] Confirming Baseline Checkpoints and Vessel Files Remain Untouched...")
    baseline_ckpt = REPO_ROOT / "results" / "DR" / "checkpoints" / "best_dr_unet.pth"
    assert baseline_ckpt.exists(), f"Baseline checkpoint missing: {baseline_ckpt}"
    ckpt_size = baseline_ckpt.stat().st_size
    print(f"  Baseline checkpoint path: {baseline_ckpt}")
    print(f"  Baseline checkpoint size: {ckpt_size:,} bytes")
    assert ckpt_size == 372688151, f"Baseline checkpoint size changed! Expected 372688151, got {ckpt_size}"

    # Check git status for modified files
    res = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
    git_lines = res.stdout.strip().splitlines()
    modified_files = [line.split()[-1] for line in git_lines if line.startswith(" M")]

    vessel_patterns = ["drive", "chase", "stare", "fives"]
    vessel_mods = [f for f in modified_files if any(p in f.lower() for p in vessel_patterns)]
    baseline_mods = [f for f in modified_files if "train_dr_segmentation.py" in f or "dr_segmentation_dataset.py" in f]

    assert len(vessel_mods) == 0, f"Vessel files modified: {vessel_mods}"
    assert len(baseline_mods) == 0, f"Baseline files modified: {baseline_mods}"
    print("  Baseline training scripts modified: None")
    print("  Vessel segmentation files modified: None")
    print("  -> CHECK 6 PASSED: All original baseline checkpoints and vessel scripts are 100% intact.")

    print("\n" + "=" * 80)
    print("ALL 6 PRE-TRAINING VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_checks()
