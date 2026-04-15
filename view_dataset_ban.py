import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def view_random_samples(dataset_path, num_samples=20):
    print(f"Đang quét thư mục: {dataset_path}")
    mask_files = glob.glob(os.path.join(dataset_path, "**", "*_mask.tif"), recursive=True)
    mask_files = [m for m in mask_files if not m.endswith("_brain_mask.tif")]
    
    if not mask_files:
        print("Không tìm thấy file mask nào.")
        return

    # Chỉ giữ các cặp có đủ image + tumor mask + brain mask
    valid_triplets = []
    print("Đang tìm các mẫu có u não...")
    for m in mask_files:
        img_path = m.replace("_mask.tif", ".tif")
        brain_mask_path = m.replace("_mask.tif", "_brain_mask.tif")
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(brain_mask_path):
            continue

        mask_img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue

        has_tumor = cv2.countNonZero(mask_img) > 0
        valid_triplets.append((m, has_tumor))

    if not valid_triplets:
        print("Không tìm thấy cặp ảnh nào có đủ image/mask/brain_mask.")
        return

    positives = [m for m, has_tumor in valid_triplets if has_tumor]
    candidates = positives if positives else [m for m, _ in valid_triplets]

    print(f"Tổng cặp hợp lệ: {len(valid_triplets)} | Có u: {len(positives)} | Đang hiển thị: {min(num_samples, len(candidates))}")
    samples = random.sample(candidates, min(num_samples, len(candidates)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3.2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, mask_path in enumerate(samples):
        img_path = mask_path.replace("_mask.tif", ".tif")
        brain_mask_path = mask_path.replace("_mask.tif", "_brain_mask.tif")
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        brain_mask = np.zeros_like(mask)
        if os.path.exists(brain_mask_path):
            brain_mask = cv2.imread(brain_mask_path, cv2.IMREAD_GRAYSCALE)
            
        ax_img, ax_brain, ax_tumor, ax_overlay = axes[i]
        
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Image\n{os.path.basename(img_path)}")
        ax_img.axis('off')
        
        ax_brain.imshow(brain_mask, cmap='gray')
        ax_brain.set_title("Brain Mask")
        ax_brain.axis('off')
        
        ax_tumor.imshow(mask, cmap='gray')
        ax_tumor.set_title("Tumor Mask")
        ax_tumor.axis('off')
        
        # Overlay
        overlay = img.copy()
        overlay[brain_mask > 0] = overlay[brain_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5 # Xanh dương
        overlay[mask > 0]       = overlay[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5       # Đỏ
        
        ax_overlay.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax_overlay.set_title("Overlay (Xanh: Não, Đỏ: U)")
        ax_overlay.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "lgg-mri-segmentation", "kaggle_3m")
    view_random_samples(dataset_dir, num_samples=20)
