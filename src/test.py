from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

# Carica modello e processor
model_name = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/segformer"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# Mappa id → label
id2label = model.config.id2label

def segment_image(image_path, output_dir="output_segments", conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    # Carica immagine
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Upsampling logits a dimensione originale
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(orig_height, orig_width),  # (H, W)
        mode="bilinear",
        align_corners=False
    )

    # Softmax → probabilità per ogni pixel
    probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]  # (num_classes, H, W)

    # Mappa segmenti (argmax)
    seg = torch.argmax(probs, dim=0).cpu().numpy()

    # Classi uniche senza background
    unique_segments = np.unique(seg)
    unique_segments = unique_segments[unique_segments != 0]

    # Funzione posizione
    def get_position_name(mask):
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            return "unknown"
        cx, cy = xs.mean(), ys.mean()
        if cx < orig_width / 3:
            horizontal_pos = "left"
        elif cx < 2 * orig_width / 3:
            horizontal_pos = "center"
        else:
            horizontal_pos = "right"
        if cy < orig_height / 3:
            vertical_pos = "top"
        elif cy < 2 * orig_height / 3:
            vertical_pos = "middle"
        else:
            vertical_pos = "bottom"
        return f"{vertical_pos} {horizontal_pos}" if vertical_pos != "middle" else horizontal_pos

    # Font per testo
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    # Overlay
    overlay = np.array(image).copy()
    kept_segments = 0

    for seg_id in unique_segments:
        mask = (seg == seg_id).astype(np.uint8)

        # Confidenza media del segmento
        seg_conf = probs[seg_id][mask == 1].mean().item()

        if seg_conf < conf_threshold:
            continue  # scarta segmenti poco sicuri

        kept_segments += 1
        position_name = get_position_name(mask)
        class_name = id2label.get(int(seg_id), f"class_{seg_id}")
        label_text = f"{class_name} - {position_name} ({seg_conf:.2f})"

        # Immagine RGBA grande come originale
        img_np = np.array(image)
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack((img_np, alpha))
        seg_img = Image.fromarray(rgba, mode="RGBA")

        # Aggiungi testo
        draw_seg = ImageDraw.Draw(seg_img)
        draw_seg.text((10, 10), label_text, fill=(255, 0, 0, 255), font=font)
        seg_img.save(os.path.join(output_dir, f"{class_name}_{position_name}.png"))

        # Colore casuale per overlay
        color = [random.randint(0, 255) for _ in range(3)]
        overlay[mask == 1] = (np.array(color) * 0.5 + overlay[mask == 1] * 0.5).astype(np.uint8)

        # Aggiungi testo su overlay
        ys, xs = np.where(mask == 1)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            overlay_pil = Image.fromarray(overlay)
            draw_overlay = ImageDraw.Draw(overlay_pil)
            draw_overlay.text((cx, cy), class_name, fill=(255, 255, 255), font=font)
            overlay = np.array(overlay_pil)

    # Salva overlay
    Image.fromarray(overlay).save(os.path.join(output_dir, "segments_overlay.png"))

    print(f"Saved {kept_segments} full-size labeled segments (conf > {conf_threshold}) and overlay to '{output_dir}'.")


if __name__ == "__main__":
    segment_image(
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/colosseo.jpg",
        output_dir="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/img_segm",
        conf_threshold=0.6 
    )
