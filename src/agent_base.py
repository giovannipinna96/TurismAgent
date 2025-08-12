from smolagents import Tool, CodeAgent, TransformersModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, SegformerForSemanticSegmentation, AutoImageProcessor
import numpy as np
import os
from geoclip import GeoCLIP
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


# ==== STEP 1: Setup CLIP ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/clip-vit-base-patch32"
).to(device)
processor = CLIPProcessor.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/clip-vit-base-patch32"
)

streetclip_model = CLIPModel.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
)
streetclip_processor = CLIPProcessor.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
)

model_name = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/segformer"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.to(device)

# Mappa id → label
id2label = model.config.id2label


# GeoCLIP
geoclip_model = GeoCLIP().to(device)

# Geopy per conversione coordinate <-> stato
geolocator = Nominatim(user_agent="LocalizatorGeoAgent")

# Lista label StreetCLIP
labels = [
    "Albania", "Andorra", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bermuda", "Bhutan", "Bolivia", "Botswana", "Brazil", "Bulgaria", "Cambodia", "Canada",
    "Chile", "China", "Colombia", "Croatia", "Czech Republic", "Denmark", "Dominican Republic",
    "Ecuador", "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Greenland", "Guam",
    "Guatemala", "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan",
    "Jordan", "Kenya", "Kyrgyzstan", "Laos", "Latvia", "Lesotho", "Lithuania", "Luxembourg",
    "Macedonia", "Madagascar", "Malaysia", "Malta", "Mexico", "Monaco", "Mongolia", "Montenegro",
    "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Palestine", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Romania", "Russia", "Rwanda", "Senegal", "Serbia", "Singapore",
    "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Swaziland", "Sweden",
    "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay"
]



# ==== STEP 2: Database in memoria ====
image_db = []  # Lista di dict: {path, embedding, text}


def embed_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()


# ==== STEP 3: Download immagini monumenti ====
monuments = [
    (
        "big_ben.png",
        "Big Ben",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg",
    ),
    (
        "statue_liberty.png",
        "Statua della Libertà",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/statua_liberta.jpg",
    ),
    (
        "tour_eiffel.png",
        "Tour Eiffel",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/tour_eiffel.jpg",
    ),
    (
        "colosseo.png",
        "Colosseo",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/colosseo.jpg",
    ),
    (
        "cristo_redentor.png",
        "Cristo Redentore",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/cristo_redentor.jpg",
    ),
]

for filename, name, url in monuments:
    img = Image.open(url).convert("RGB")
    # img.save(filename)
    emb = embed_image(img)
    image_db.append({"path": filename, "embedding": emb, "text": name})


# ==== STEP 4: Definizione Tool ====
# class ImageRetrievalTool(Tool):
#     name = "image_retrieval"
#     description = "Give a path of an image find the name of the building and description of the building inside that image"
#     inputs = {
#         "image_path": {"type": "string", "description": "Path of the image to analyze."}
#     }
#     output_type = "string"

#     def forward(self, image_path: str) -> str:
#         query_img = Image.open(image_path).convert("RGB")
#         query_emb = embed_image(query_img)

#         sims = []
#         for entry in image_db:
#             sim = np.dot(query_emb, entry["embedding"]) / (
#                 np.linalg.norm(query_emb) * np.linalg.norm(entry["embedding"])
#             )
#             sims.append((sim, entry["text"]))

#         sims.sort(reverse=True, key=lambda x: x[0])
#         top1 = [name for _, name in sims[:1]]

#         return f"Name: {', '.join(top1)}"

class ImageRetrievalTool(Tool):
    name = "image_retrieval"
    description = "Given a path of an image, segment it and find the name of the building for each detected segment with similarity above threshold."
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to analyze."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        output_dir = "tmp_segments"
        os.makedirs(output_dir, exist_ok=True)
        segments_info = self.segment_image_and_get_paths(image_path, output_dir, conf_threshold=0.6)

        results_str = []
        for seg_path, label_text, seg_conf in segments_info:
            query_img = Image.open(seg_path).convert("RGB")
            query_emb = embed_image(query_img)

            sims = []
            for entry in image_db:
                sim = np.dot(query_emb, entry["embedding"]) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(entry["embedding"])
                )
                if sim >= 0.6:
                    sims.append((sim, entry["text"]))

            sims.sort(reverse=True, key=lambda x: x[0])

            # Crea descrizione dettagliata per questo segmento
            match_desc = (
                ", ".join([f"{name} ({score:.2f})" for score, name in sims])
                if sims else "No matches above similarity threshold"
            )

            segment_report = (
                f"Segment Path: {seg_path}\n"
                f"Description: {label_text}\n"
                f"Confidence: {seg_conf:.2f}\n"
                f"Matches: {match_desc}\n"
                "----------------------------------------"
            )
            results_str.append(segment_report)

        return "\n".join(results_str)

    def segment_image_and_get_paths(self, image_path, output_dir="output_segments", conf_threshold=0.5):
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False
        )
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
        seg = torch.argmax(probs, dim=0).cpu().numpy()
        unique_segments = np.unique(seg)
        unique_segments = unique_segments[unique_segments != 0]

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

        segment_paths = []
        for seg_id in unique_segments:
            mask = (seg == seg_id).astype(np.uint8)
            seg_conf = probs[seg_id][mask == 1].mean().item()
            if seg_conf < conf_threshold:
                continue

            position_name = get_position_name(mask)
            class_name = id2label.get(int(seg_id), f"class_{seg_id}")
            label_text = f"{class_name} - {position_name}"

            img_np = np.array(image)
            alpha = (mask * 255).astype(np.uint8)
            rgba = np.dstack((img_np, alpha))
            seg_img = Image.fromarray(rgba, mode="RGBA")

            seg_path = os.path.join(output_dir, f"{class_name}_{position_name}.png")
            seg_img.save(seg_path)
            segment_paths.append((seg_path, label_text, seg_conf))

        return segment_paths



# class Localizator(Tool):
#     name = "localizator"
#     description = "Tool for localizing an image in a country giving as input a image path."
#     inputs = {
#         "image_path": {
#             "type": "string",
#             "description": "Path of the image to geolocalize.",
#         }
#     }
#     output_type = "string"

#     def forward(self, image_path: str) -> str:

#         query_img = Image.open(image_path).convert("RGB")
#         inputs = streetclip_processor(
#             text=labels, images=query_img, return_tensors="pt", padding=True
#         )
#         with torch.no_grad():
#             outputs = streetclip_model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         prediction = logits_per_image.softmax(dim=1)
#         confidences = {
#             labels[i]: float(prediction[0][i].item()) for i in range(len(labels))
#         }

#         sorted_confidences = sorted(
#             confidences.items(), key=lambda item: item[1], reverse=True
#         )
#         top_label, top_confidence = sorted_confidences[0]
#         return f"Country: {top_label}"


class Localizator(Tool):
    name = "localizator"
    description = "Localizza un'immagine usando sia GeoCLIP che StreetCLIP, restituendo latitudine, longitudine, stato e città."
    inputs = {
        "image_path": {"type": "string", "description": "Path dell'immagine da geolocalizzare."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")

        # ==== Metodo 1: StreetCLIP ====
        street_inputs = streetclip_processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            street_outputs = streetclip_model(**street_inputs)
        logits_per_image = street_outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)
        sorted_confidences = sorted(
            {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}.items(),
            key=lambda item: item[1], reverse=True
        )
        street_country, street_conf = sorted_confidences[0]
        street_lat, street_lon = self.get_country_coordinates(street_country)
        street_city, street_country_name = self.reverse_geocode(street_lat, street_lon)

        # ==== Metodo 2: GeoCLIP ====
        top_pred_gps, _ = geoclip_model.predict(image_path, top_k=1)
        geo_lat, geo_lon = top_pred_gps[0]
        geo_city, geo_country = self.reverse_geocode(geo_lat, geo_lon)

        # ==== Output fancy ====
        return (
            "🌍 **Risultati Localizzazione** 🌍\n"
            f"📌 **StreetCLIP**: {street_city}, {street_country_name} ({street_conf*100:.2f}%)\n"
            f"   Lat: {street_lat:.6f}, Lon: {street_lon:.6f}\n\n"
            f"📌 **GeoCLIP**: {geo_city}, {geo_country}\n"
            f"   Lat: {geo_lat:.6f}, Lon: {geo_lon:.6f}"
        )

    def get_country_coordinates(self, country_name):
        try:
            location = geolocator.geocode(country_name, timeout=10)
            if location:
                return location.latitude, location.longitude
        except GeocoderTimedOut:
            return (0.0, 0.0)
        return (0.0, 0.0)

    def reverse_geocode(self, lat, lon):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
            if location and location.raw.get("address"):
                city = location.raw["address"].get("city") or location.raw["address"].get("town") or location.raw["address"].get("village") or "Unknown"
                country = location.raw["address"].get("country", "Unknown")
                return city, country
        except GeocoderTimedOut:
            return "Unknown", "Unknown"
        return "Unknown", "Unknown"


# ==== STEP 5: Agente smolagents ====
retrieval_tool = ImageRetrievalTool()
localizator = Localizator()
# agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/TinyAgent-1.1B", trust_remote_code=True, device_map="auto")
agent_model = TransformersModel(
    model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True,
    device_map="auto",
)
# agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen3-4B-Thinking-2507", trust_remote_code=True, device_map="auto")

agent = CodeAgent(
    tools=[localizator, retrieval_tool],
    model=agent_model,
    additional_authorized_imports=["PIL", "torch", "transformers", "numpy", "io"],
    planning_interval=3,  # Esegui il planning ogni 1 step
)

# ==== STEP 6: Test ====
file_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/img_segm/building_center.png"
result = agent.run(f"Localize the image: {file_path} and then describe it.")
print(result)
