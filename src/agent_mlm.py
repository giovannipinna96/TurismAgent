import logging
from typing import Dict
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    Mask2FormerForUniversalSegmentation, AutoModelForVisualQuestionAnswering,
    BlipProcessor, BlipForQuestionAnswering
    )
from smolagents import Tool, CodeAgent, TransformersModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifierTool(Tool):
    name = "image_classifier"
    description = (
        "The tool classifies an image."
        "Returns the predicted class and confidence."
    )
    inputs = {
        'question': {'type': "string", 'description': "Question about the image"},
        "image_path": {"type": "string", "description": "Path to the image file"}
    }
    output_type = "string"

    def forward(self, question: str, image_path: str) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/blip-vqa-base", local_files_only=True)
        model = BlipForQuestionAnswering.from_pretrained("/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/blip-vqa-base",
                                                         local_files_only=True).to(device)
        image_path = Path(image_path)
        if not image_path.exists():
            return f"Error: image file not found at {image_path}"

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        answer = processor.decode(generated_ids[0], skip_special_tokens=True)
        return f"Answer: {answer}"


def create_agent():
    classifier_tool = ImageClassifierTool()
    # The agent model can be a local LLM or a small orchestrator model
    # agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/gpt-oss-20B", trust_remote_code=True, device_map="auto")
    agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen3-4B-Thinking-2507", trust_remote_code=True, device_map="auto")
    agent = CodeAgent(
        tools=[classifier_tool],
        model=agent_model,
        additional_authorized_imports=["PIL", "torch", "transformers", 'pathlib']
    )
    return agent

if __name__ == "__main__":
    # Use the ViT base model from Hugging Face (download once, then use offline with local_files_only=True)
    # local_model_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/vit-base"
    agent = create_agent()
    image_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg"  # path immagine locale
    result = agent.run(
        f"What there in this image {image_path}?",
    )
    print(result)