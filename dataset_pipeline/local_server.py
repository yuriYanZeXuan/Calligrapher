import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import uvicorn

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Static Directory for Images ---
STATIC_DIR = "generated_images"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name="static")


# --- OpenAI Compatible Pydantic Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen-image-edit"
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    extra_body: Optional[Dict[str, Any]] = None

class ImageURL(BaseModel):
    url: str

class ImageGenerationResponse(BaseModel):
    data: List[ImageURL]

# --- Placeholder for the actual model ---
def placeholder_image_generator(prompt: str, size: tuple[int, int], save_path: str):
    """
    This is a placeholder function.
    Replace this with your actual QwenImageEdit model call.
    It currently generates a gray image with the prompt text written on it.
    """
    try:
        image = Image.new('RGB', size, color = 'gray')
        draw = ImageDraw.Draw(image)
        
        # Use a basic font
        try:
            font = ImageFont.truetype("Arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()

        # Simple text wrapping
        lines = []
        words = prompt.split()
        current_line = ""
        for word in words:
            if font.getsize(current_line + word)[0] < size[0] - 20:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        y_text = 50
        for line in lines:
            draw.text((10, y_text), line.strip(), fill='white', font=font)
            y_text += font.getsize(line)[1]

        image.save(save_path)
        print(f"Placeholder image saved to {save_path}")
    except Exception as e:
        print(f"Error in placeholder_image_generator: {e}")
        # Create a blank image on error to avoid breaking the flow
        Image.new('RGB', size, color = 'red').save(save_path)


# --- API Endpoint ---
@app.post("/v1/images/generate", response_model=ImageGenerationResponse)
async def generate_image_endpoint(request: ImageGenerationRequest, http_request: Request):
    """
    OpenAI-compatible endpoint for image generation.
    """
    print(f"Received image generation request with prompt: '{request.prompt}'")
    
    # --- Model Inference (Placeholder) ---
    # In a real implementation, you would call your QwenImageEdit model here.
    # The model would take the `request.prompt` and other parameters.
    
    image_filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(STATIC_DIR, image_filename)
    
    # Parse size
    try:
        width, height = map(int, request.size.split('x'))
    except:
        width, height = 1024, 1024 # Default size

    # This is where you would replace the placeholder call
    placeholder_image_generator(request.prompt, (width, height), output_path)
    
    # --- Construct Response ---
    # Get the base URL (e.g., http://127.0.0.1:8000)
    base_url = str(http_request.base_url)
    image_url = f"{base_url}{STATIC_DIR}/{image_filename}"
    
    response_data = ImageGenerationResponse(data=[ImageURL(url=image_url)])
    
    return JSONResponse(content=response_data.dict())

if __name__ == "__main__":
    print("Starting local OpenAI-compatible server on http://127.0.0.1:8000")
    print("Press CTRL+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
