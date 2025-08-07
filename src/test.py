# from transformers import pipeline
# from PIL import Image
# import os

# # Carica un'immagine di esempio (puoi sostituire con un'immagine locale)
# image_url = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg"
# image = Image.open(image_url).convert("RGB")
# # Crea pipeline con il modello panottico
# panoptic_pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

# # Esegui segmentazione
# outputs = panoptic_pipeline(image)

# # Crea una cartella per salvare i risultati
# output_dir = "segmentazione_output"
# os.makedirs(output_dir, exist_ok=True)

# # Salva immagine originale
# image.save(os.path.join(output_dir, "immagine_originale.jpg"))
# print("✅ Immagine originale salvata.")

# # Salva ogni maschera individuale
# for idx, segment in enumerate(outputs):
#     label = segment['label'].replace(" ", "_")
#     score = segment['score']
#     mask = segment['mask']  # È una PIL.Image
    
#     filename = f"maschera_{idx+1:02d}_{label}_{score:.2f}.png"
#     mask_path = os.path.join(output_dir, filename)
#     mask.save(mask_path)
#     print(f"✅ Maschera salvata: {filename}")

# print("\n🎉 Segmentazione completata. Maschere salvate in:", output_dir)


from transformers import pipeline
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Percorso immagine locale
image_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg"
image = Image.open(image_path).convert("RGB")

# Crea pipeline SAM
segmenter = pipeline("mask-generation", model="facebook/sam-vit-huge", use_fast=True)

# Punto di input (coordinata x, y) – modifica secondo la tua immagine
# input_point = [[500, 400]]   # <-- Cambia queste coordinate a piacere
input_point = [[[450, 600]]]
input_label = [1]            # 1 = foreground

# Segmentazione
masks = segmenter(image, input_points=input_point, input_labels=input_label)

# Cartella di output
output_dir = "maschere_sam"
os.makedirs(output_dir, exist_ok=True)

# # Salva tutte le maschere individuali (opzionale)
# for i in range(min(10, masks.shape[0])):
#     single_mask = masks[i][0].mul(255).byte().cpu().numpy()
#     Image.fromarray(single_mask).save(os.path.join(output_dir, f"maschera_{i+1:02d}.png"))

# ✅ Funzione per disegnare maschera colorata (come da tuo codice)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

plt.imshow(np.array(image))
ax = plt.gca()
for mask in masks["masks"]:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
# ✅ Salva immagine finale con maschere sovrapposte
final_output_path = os.path.join(output_dir, "immagine_con_maschere.png")
plt.savefig(final_output_path, bbox_inches='tight', pad_inches=0)
print(f"🎉 Immagine finale salvata in: {final_output_path}")