import re
from PIL import Image
import numpy as np
import os


def parse_generated_filename(filename):
    """
    Parses the generated filename to extract metadata.
    Example filename: result_0_test1_The_text_is_Calligrapher._12345678.png
    """
    # Use a more robust, non-greedy regex to capture the prompt, which may contain special characters.
    match = re.search(r'result_\d+_(test\d+)_(.*?)_(\d+)\.png', os.path.basename(filename))
    if not match:
        return None

    ref_id_str = match.group(1)
    safe_prompt = match.group(2)
    seed = int(match.group(3))

    # Reconstruct the original prompt (this is an approximation)
    prompt_text = safe_prompt.replace('_', ' ').replace("The text is ", "").strip()

    return {
        'ref_id': ref_id_str,
        'prompt': prompt_text,
        'seed': seed
    }


def load_images_for_evaluation(generated_path, benchmark_dir):
    """
    Loads all necessary images for evaluation based on the generated image path.
    """
    metadata = parse_generated_filename(generated_path)
    if not metadata:
        raise ValueError(f"Could not parse filename: {generated_path}")

    ref_id = metadata['ref_id']

    source_path = os.path.join(benchmark_dir, f"{ref_id}_source.png")
    mask_path = os.path.join(benchmark_dir, f"{ref_id}_mask.png")
    ref_path = os.path.join(benchmark_dir, f"{ref_id}_ref.png")

    try:
        generated_img = Image.open(generated_path).convert("RGB")
        source_img = Image.open(source_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")  # Grayscale
        ref_img = Image.open(ref_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error loading files for {ref_id}: {e}")
        return None, None, None, None, None

    # Ensure mask is binary (0 or 255)
    mask_np = np.array(mask_img)
    mask_np[mask_np > 0] = 255
    mask_img = Image.fromarray(mask_np)

    return generated_img, source_img, mask_img, ref_img, metadata
