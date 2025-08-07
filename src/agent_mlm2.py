# agent_main.py

from smolagents import Tool, CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch


# class ImageQuestionTool(Tool):
#     name = "image_question_tool"
#     description = "Risponde a domande su un'immagine usando un modello multimodale."
#     inputs = {
#         "image_path": {"type": "str", "description": "Percorso locale dell'immagine"},
#         "question": {"type": "str", "description": "Domanda da porre sull'immagine"}
#     }
#     output_type = "str"

#     def __init__(self, model_name="Qwen/Qwen-VL-Chat"):
#         super().__init__()
#         self.model_name = model_name
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None,
#         )

#     def forward(self, image_path: str, question: str) -> str:
#         image = Image.open(image_path).convert("RGB")

#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": question}
#                 ]
#             }
#         ]

#         inputs = self.processor.apply_chat_template(
#             conversation,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.model.device)

#         with torch.no_grad():
#             output = self.model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 temperature=0.7,
#                 pad_token_id=self.processor.tokenizer.eos_token_id
#             )

#         response = self.processor.tokenizer.decode(
#             output[0][inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         )

#         return response.strip()


# ==== AGENTE ====

def create_image_agent():
    # tools = [ImageQuestionTool()]
    tools = []
    # model = TransformersModel(model_id="Qwen/Qwen-VL-Chat", trust_remote_code=True, device_map="auto")
    model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/qwen-vl", trust_remote_code=True, device_map="auto")
    agent = CodeAgent(
        tools=tools,
        additional_authorized_imports=["PIL", "torch", "transformers"],
        model=model,  # puoi usare anche GPT qui per orchestrazione se vuoi
    )
    return agent


# ==== ESEMPIO DI UTILIZZO ====

if __name__ == "__main__":
    image_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg"  # path immagine locale
    question = "Describe this image. What monument or objects are visible?"

    agent = create_image_agent()
    result = agent.run(
        f"Usa lo strumento per analizzare '{image_path}' e rispondere alla domanda: '{question}'"
    )
    print("\n--- RISPOSTA AGENTE ---\n")
    print(result)
