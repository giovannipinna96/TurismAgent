
from smolagents import CodeAgent, TransformersModel

from PIL import Image, ImageDraw
import re


def draw_bounding_boxes(image, bounding_boxes, outline_color="red", line_width=2):
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image

def rescale_bounding_boxes(bounding_boxes, original_width, original_height, scaled_width=1000, scaled_height=1000):
    x_scale = original_width / scaled_width
    y_scale = original_height / scaled_height
    rescaled_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rescaled_box = [
            xmin * x_scale,
            ymin * y_scale,
            xmax * x_scale,
            ymax * y_scale
        ]
        rescaled_boxes.append(rescaled_box)
    return rescaled_boxes

default_system_prompt = """You are a helpfull assistant to detect objects in images. When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] whith the values beeing scaled to 1000 by 1000 pixels. When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."""


def create_image_agent():
    # Use the Qwen2-VL-7B-Instruct model from local path with InferenceClientModel for smolagents
    model_id = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/qwen-vl"
    model = TransformersModel(model_id=model_id)
    agent = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=["PIL", "torch", "transformers", "base64", "io", "re"],
        instructions=default_system_prompt
    )
    return agent


# ==== ESEMPIO DI UTILIZZO ====



if __name__ == "__main__":
    image_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg"  # path immagine locale
    user_prompt = "detect all objects in the image"

    # Load image
    image = Image.open(image_path).convert("RGB")
    agent = create_image_agent()

    # Compose the agent's input prompt
    prompt = f"Given the image at path '{image_path}', {user_prompt}. Return the bounding boxes as described in the system prompt."
    output_text = agent.run(prompt)

    print("\n--- RAW AGENT OUTPUT ---\n")
    print(output_text)

    # Extract bounding boxes from output
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, str(output_text))
    parsed_boxes = [[int(num) for num in match] for match in matches]
    scaled_boxes = rescale_bounding_boxes(parsed_boxes, image.width, image.height)

    # Draw bounding boxes
    annotated_image = draw_bounding_boxes(image.copy(), scaled_boxes)

    print("\n--- PARSED BOUNDING BOXES ---\n")
    print(parsed_boxes)
    print("\n--- SCALED BOUNDING BOXES (image coordinates) ---\n")
    print(scaled_boxes)
    annotated_image.save("annotated_output.png")
    print("\nAnnotated image saved as annotated_output.png")
