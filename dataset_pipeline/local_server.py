import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import argparse
import torch
from diffusers import QwenImagePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

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

# --- OpenAI Compatible Models for Chat ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen2.5llm"
    messages: List[ChatMessage]
    # Add other params like temperature if needed

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(uuid.uuid4().int >> 64))
    model: str
    choices: List[ChatCompletionChoice]

# ==============================================================================
# --- Global Model Holder ---
# ==============================================================================
class ModelHolder:
    image_pipeline = None
    llm_model = None
    tokenizer = None

# ==============================================================================
# --- FastAPI Events ---
# ==============================================================================
async def load_image_model():
    """Load the QwenImagePipeline model."""
    print("Loading Image Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = config.SERVICES["local"]["image_model_path"]
    
    ModelHolder.image_pipeline = QwenImagePipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16
    ).to(device)
    print(f"Image Model loaded from {model_path}.")

async def load_llm_model():
    """Load the Qwen LLM and tokenizer."""
    print("Loading Language Model...")
    model_path = config.SERVICES["local"]["llm_model_path"]
    
    ModelHolder.tokenizer = AutoTokenizer.from_pretrained(model_path)
    ModelHolder.llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Language Model loaded from {model_path}.")

# ==============================================================================
# --- Model Implementation ---
# ==============================================================================
def image_generator(prompt: str, size: tuple[int, int], save_path: str):
    """
    [!!!] This is the REAL implementation using the QwenImagePipeline.
    
    This function takes a text prompt and saves a generated image to `save_path`.
    """
    if ModelHolder.image_pipeline is None:
        raise RuntimeError("Image generation pipeline is not initialized.")
    
    try:
        print(f"Generating image for prompt: '{prompt}'")
        # The pipeline returns a list of images, we take the first one
        image = ModelHolder.image_pipeline(
            prompt=prompt,
            negative_prompt=" ", # A negative prompt is often recommended
            height=size[1],
            width=size[0],
            num_inference_steps=50,
        ).images[0]
        
        image.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error in image_generator: {e}")
        # Create a blank image on error to avoid breaking the flow
        Image.new('RGB', size, color='red').save(save_path)

def llm_generator(messages: List[Dict]) -> str:
    """
    [!!!] This is the REAL implementation using Qwen2.5-7B-Instruct.
    
    This function takes a list of messages and returns a text response.
    """
    if not ModelHolder.llm_model or not ModelHolder.tokenizer:
        raise RuntimeError("LLM model or tokenizer is not initialized.")
        
    try:
        # Apply the chat template to format the conversation
        text = ModelHolder.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the formatted text
        model_inputs = ModelHolder.tokenizer([text], return_tensors="pt").to(ModelHolder.llm_model.device)
        
        # Generate a response
        generated_ids = ModelHolder.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512 # Adjust as needed
        )
        
        # Decode the response, skipping the prompt tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = ModelHolder.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()

    except Exception as e:
        print(f"Error in llm_generator: {e}")
        return "error_in_llm"

# Define endpoint functions without decorators
async def generate_image_endpoint(request: ImageGenerationRequest, http_request: Request):
    """OpenAI-compatible endpoint for image generation."""
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

    # ====================================================================
    # This is where the real model is called
    image_generator(request.prompt, (width, height), output_path)
    # ====================================================================
    
    # --- Construct Response ---
    # Get the base URL (e.g., http://127.0.0.1:8000)
    base_url = str(http_request.base_url)
    image_url = f"{base_url}{STATIC_DIR}/{image_filename}"
    
    response_data = ImageGenerationResponse(data=[ImageURL(url=image_url)])
    
    return JSONResponse(content=response_data.dict())

async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible endpoint for chat completions."""
    print(f"Received chat completion request for model: '{request.model}'")
    
    # ====================================================================
    # Replace this with your actual Qwen LLM call
    messages_dict = [msg.dict() for msg in request.messages]
    simplified_text = llm_generator(messages_dict)
    # ====================================================================
    
    # --- Construct Response ---
    response_message = ChatMessage(role="assistant", content=simplified_text)
    choice = ChatCompletionChoice(message=response_message)
    response = ChatCompletionResponse(model=request.model, choices=[choice])
    
    return JSONResponse(content=response.dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local model microservice.")
    parser.add_argument(
        "--service", 
        type=str, 
        required=True,
        choices=["image", "llm"],
        help="Specify the service to run: 'image' or 'llm'."
    )
    parser.add_argument(
        "--port", 
        type=int,
        help="Specify the port to run the service on. Defaults to 8000 for image, 8001 for llm."
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU device IDs to use (e.g., '0,1,2,3')."
    )
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES before importing torch or diffusers
    if args.device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    # Determine default port if not provided
    if args.port is None:
        args.port = 8000 if args.service == "image" else 8001

    # Register startup event and endpoints based on service type
    if args.service == "image":
        app.add_event_handler("startup", load_image_model)
        # Match the endpoint that the OpenAI client library seems to be incorrectly calling
        app.add_api_route("/v1/images/generations", generate_image_endpoint, methods=["POST"])
        print("Starting local IMAGE generation server...")
    elif args.service == "llm":
        app.add_event_handler("startup", load_llm_model)
        app.add_api_route("/v1/chat/completions", create_chat_completion, methods=["POST"])
        print("Starting local LLM server...")

    device_info = f"on GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}" if torch.cuda.is_available() else "on CPU"
    print(f"Service '{args.service}' will run on http://127.0.0.1:{args.port} {device_info}")
    print("Press CTRL+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
