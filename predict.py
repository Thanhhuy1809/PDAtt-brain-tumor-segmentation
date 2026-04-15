import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ===== LOAD MODEL =====
def load_model(checkpoint_path, device):
    import Architectures as networks

    model = networks.PYAttUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ===== LOAD IMAGE =====
def load_image(path, img_size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(img).unsqueeze(0)  # (1,3,H,W)


# ===== PREDICT =====
def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        out_tumor, _ = model(image_tensor)
        prob = torch.sigmoid(out_tumor)
        pred = (prob > 0.5).float()

    return prob.cpu().numpy()[0, 0], pred.cpu().numpy()[0, 0]


# ===== VISUALIZE =====
def show_result(image_path, prob, pred):
    img = np.array(Image.open(image_path).convert("RGB"))

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Probability")
    plt.imshow(prob, cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Prediction Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.show()


# ===== MAIN =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt-Unet-main2\Dataset_Ban\outputs\pdatt_dataset_ban\best.pt", device)

    image_path = "path_to_your_image.tif"  # sửa đường dẫn

    img_tensor = load_image(image_path)
    prob, pred = predict(model, img_tensor, device)

    show_result(image_path, prob, pred)