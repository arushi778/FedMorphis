"""
Audit of the 6 validation prediction panels in results/DR/validation_predictions/
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image
import numpy as np
import torch
from dr_segmentation_dataset import DDRLesionDataset, LESION_TYPES
from shared_unet import SharedUNet
from evaluate_dr_segmentation import compute_metrics

def main():
    repo_root = REPO_ROOT
    dataset = DDRLesionDataset(root_dir=repo_root / "data" / "DR" / "DDR-dataset" / "lesion_segmentation", split="valid", image_size=512)
    ckpt_path = repo_root / "results" / "DR" / "checkpoints" / "best_dr_unet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SharedUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    panel_ids = [
        "007-5730-300",
        "007-5797-300",
        "007-5800-300",
        "007-5803-300",
        "007-5831-300",
        "007-5832-300",
    ]

    print("=" * 80)
    print("AUDIT OF THE 6 VALIDATION PREDICTION PANELS")
    print("=" * 80)

    for rank, img_id in enumerate(panel_ids, 1):
        idx = None
        for i in range(len(dataset)):
            if dataset.get_sample_info(i)["image_id"] == img_id:
                idx = i
                break

        info = dataset.get_sample_info(idx)
        img_t, mask_t = dataset[idx]

        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device)).cpu()
            probs = torch.sigmoid(logits)

        m = compute_metrics(logits, mask_t.unsqueeze(0), threshold=0.5)

        # Check raw image and masks
        raw_img = Image.open(info["image_path"])
        raw_w, raw_h = raw_img.size

        raw_lesions = {}
        for l in LESION_TYPES:
            m_path = info["mask_paths"][l]
            if m_path.exists():
                arr = np.array(Image.open(m_path))
                pos = int(np.sum(arr > 0))
                if pos > 0:
                    raw_lesions[l] = pos

        gt_px = int(mask_t.sum().item())
        pred_px = int((probs > 0.5).sum().item())
        total_px = 512 * 512

        print(f"Panel #{rank}: Image {img_id} (Dataset Index: {idx})")
        print(f"  Raw Resolution: {raw_w} x {raw_h} (Aspect Ratio: {raw_w/raw_h:.2f})")
        print(f"  Lesion types present at raw resolution: {raw_lesions}")
        print(f"  512x512 GT lesion px:    {gt_px:5d} ({gt_px / total_px * 100:.3f}% of image)")
        print(f"  512x512 Pred lesion px:  {pred_px:5d} ({pred_px / total_px * 100:.3f}% of image)")
        print(f"  Prediction / GT Ratio:   {pred_px / max(gt_px, 1):.3f}")
        print(f"  Metrics (@ thr=0.5):")
        print(f"    Dice:        {m['dice']:.4f}")
        print(f"    IoU:         {m['iou']:.4f}")
        print(f"    Sensitivity: {m['sensitivity']:.4f}")
        print(f"    Specificity: {m['specificity']:.4f}")
        print(f"  Pixel Counts:")
        print(f"    TP: {m['tp']:5d}  |  FP: {m['fp']:5d}")
        print(f"    FN: {m['fn']:5d}  |  TN: {m['tn']:5d}")
        print("-" * 80)

if __name__ == "__main__":
    main()
