import os
import json
import uuid
import argparse
import config
from step1_generate_image import generate_image
from step2_simplify_prompt import simplify_prompt
from step3_ocr import ocr_image,ocr_image_paddle

def create_dataset_entry(human_written_instruction: str, service: str):
    """
    Runs the full pipeline for a single instruction to create a dataset entry.
    """
    print(f"--- Starting pipeline for instruction: '{human_written_instruction}' using '{service}' service ---")
    
    # In a real pipeline, you would have a more sophisticated prompt augmentation step
    augmented_prompt = human_written_instruction 
    
    # === Step 1: Generate Image ===
    unique_id = uuid.uuid4()
    image_filename = f"{unique_id}.png"
    image_path = os.path.join(config.OUTPUT_DIR, image_filename)
    
    if not generate_image(augmented_prompt, image_path, service=service):
        print("--- Pipeline failed at Step 1: Image Generation ---")
        return

    # === Step 2: Simplify Prompt ===
    # This is the reference text we expect to find in the image.
    reference_text = simplify_prompt(augmented_prompt, service=service)
    if not reference_text:
        print("--- Pipeline failed at Step 2: Simplify Prompt ---")
        return

    # === Step 3: OCR ===
    ocr_results = ocr_image_paddle(image_path)
    if not ocr_results:
        print("--- Pipeline failed at Step 3: OCR ---")
        # Decide if you want to keep images where OCR failed. For now, we'll discard.
        os.remove(image_path)
        print(f"Removed image {image_path} due to OCR failure.")
        return

    # For simplicity, we'll take the first OCR result that is reasonably close to our reference text.
    # A more robust solution would involve matching algorithms (e.g., Levenshtein distance).
    matched_result = None
    for bbox, text in ocr_results:
        if reference_text.lower() in text.lower() or text.lower() in reference_text.lower():
            matched_result = {"text": text, "bounding_box": bbox}
            break
            
    if not matched_result:
        print(f"Could not find a matching text for '{reference_text}' in OCR results.")
        os.remove(image_path)
        print(f"Removed image {image_path} due to no matching text.")
        return

    # === Step 4: Create Dataset Entry ===
    dataset_entry = {
        "id": str(unique_id),
        "image_file": image_filename,
        "human_instruction": human_written_instruction,
        "augmented_prompt": augmented_prompt,
        "reference_text": reference_text,
        "ocr_result": matched_result
    }

    # Save the dataset entry as a JSON file
    json_filename = f"{unique_id}.json"
    json_path = os.path.join(config.OUTPUT_DIR, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_entry, f, indent=4, ensure_ascii=False)

    print(f"--- Pipeline completed successfully. Dataset entry saved to {json_path} ---")

def main():
    parser = argparse.ArgumentParser(description="Run the dataset generation pipeline.")
    parser.add_argument(
        '--service', 
        type=str, 
        choices=['local', 'remote'], 
        default='remote',
        help="Choose the image generation service to use ('local' or 'remote')."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Read instructions from the file
    instructions_file = os.path.join(os.path.dirname(__file__), 'instructions.txt')
    with open(instructions_file, 'r', encoding='utf-8') as f:
        instructions = [line.strip() for line in f if line.strip()]
    
    for instruction in instructions:
        create_dataset_entry(instruction, service=args.service)

if __name__ == "__main__":
    main()
