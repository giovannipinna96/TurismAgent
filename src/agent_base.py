from smolagents import Tool, CodeAgent, TransformersModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np


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
class ImageRetrievalTool(Tool):
    name = "image_retrieval"
    description = "Give a path of an image find the name of the building and description of the building inside that image"
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to analyze."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        query_img = Image.open(image_path).convert("RGB")
        query_emb = embed_image(query_img)

        sims = []
        for entry in image_db:
            sim = np.dot(query_emb, entry["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(entry["embedding"])
            )
            sims.append((sim, entry["text"]))

        sims.sort(reverse=True, key=lambda x: x[0])
        top1 = [name for _, name in sims[:1]]

        return f"Name: {', '.join(top1)}"


class Localizator(Tool):
    name = "localizator"
    description = "Tool for localizing an image in a country giving as input a image path."
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path of the image to geolocalize.",
        }
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        labels = [
            "Albania",
            "Andorra",
            "Argentina",
            "Australia",
            "Austria",
            "Bangladesh",
            "Belgium",
            "Bermuda",
            "Bhutan",
            "Bolivia",
            "Botswana",
            "Brazil",
            "Bulgaria",
            "Cambodia",
            "Canada",
            "Chile",
            "China",
            "Colombia",
            "Croatia",
            "Czech Republic",
            "Denmark",
            "Dominican Republic",
            "Ecuador",
            "Estonia",
            "Finland",
            "France",
            "Germany",
            "Ghana",
            "Greece",
            "Greenland",
            "Guam",
            "Guatemala",
            "Hungary",
            "Iceland",
            "India",
            "Indonesia",
            "Ireland",
            "Israel",
            "Italy",
            "Japan",
            "Jordan",
            "Kenya",
            "Kyrgyzstan",
            "Laos",
            "Latvia",
            "Lesotho",
            "Lithuania",
            "Luxembourg",
            "Macedonia",
            "Madagascar",
            "Malaysia",
            "Malta",
            "Mexico",
            "Monaco",
            "Mongolia",
            "Montenegro",
            "Netherlands",
            "New Zealand",
            "Nigeria",
            "Norway",
            "Pakistan",
            "Palestine",
            "Peru",
            "Philippines",
            "Poland",
            "Portugal",
            "Puerto Rico",
            "Romania",
            "Russia",
            "Rwanda",
            "Senegal",
            "Serbia",
            "Singapore",
            "Slovakia",
            "Slovenia",
            "South Africa",
            "South Korea",
            "Spain",
            "Sri Lanka",
            "Swaziland",
            "Sweden",
            "Switzerland",
            "Taiwan",
            "Thailand",
            "Tunisia",
            "Turkey",
            "Uganda",
            "Ukraine",
            "United Arab Emirates",
            "United Kingdom",
            "United States",
            "Uruguay",
        ]
        query_img = Image.open(image_path).convert("RGB")
        inputs = streetclip_processor(
            text=labels, images=query_img, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = streetclip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)
        confidences = {
            labels[i]: float(prediction[0][i].item()) for i in range(len(labels))
        }

        sorted_confidences = sorted(
            confidences.items(), key=lambda item: item[1], reverse=True
        )
        top_label, top_confidence = sorted_confidences[0]
        return f"Country: {top_label}"


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
