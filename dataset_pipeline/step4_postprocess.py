import os
import json
import shutil
from PIL import Image, ImageDraw
import config
from tqdm import tqdm


def create_benchmark_dataset(input_dir, output_dir):
    """
    Processes the intermediate dataset (images and jsons) to create a final
    benchmark dataset with source, ref, mask images and a bench txt file.

    Args:
        input_dir (str): The directory containing the intermediate .png and .json files.
        output_dir (str): The directory where the final benchmark dataset will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all json files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Error: No .json files found in {input_dir}. Please run previous steps.")
        return

    bench_txt_path = os.path.join(output_dir, "self_bench.txt")
    
    with open(bench_txt_path, 'w', encoding='utf-8') as bench_file:
        # We'll sort the files to have a consistent order
        for i, json_filename in enumerate(tqdm(sorted(json_files), desc="Processing dataset")):
            test_index = i + 1 # Benchmark files are 1-indexed
            
            base_filename = os.path.splitext(json_filename)[0]
            json_path = os.path.join(input_dir, json_filename)
            image_path = os.path.join(input_dir, f"{base_filename}.png")

            if not os.path.exists(image_path):
                print(f"Warning: Corresponding image for {json_filename} not found. Skipping.")
                continue

            # Load the original image and metadata
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                original_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error processing {json_filename}: {e}. Skipping.")
                continue

            # --- Compatibility with original main.py JSON structure ---
            # Extract text from "reference_text"
            reference_text_val = metadata.get('reference_text')
            # Extract text and bbox from the nested "ocr_result" dictionary
            ocr_result = metadata.get('ocr_result', {})
            ocr_text_val = ocr_result.get('text')
            bbox = ocr_result.get('bounding_box')
            # --- End Compatibility ---

            # --- NEW: Filtering mechanism ---
            # Skip if the reference text and the OCR'd text do not match.
            if not reference_text_val or not ocr_text_val or \
               reference_text_val.strip().lower() != ocr_text_val.strip().lower():
                print(f"\nWarning: Mismatch in {json_filename}. Skipping.")
                print(f"  - Reference: '{reference_text_val}'")
                print(f"  - OCR Text : '{ocr_text_val}'")
                continue
            # --- End Filtering ---
            
            text = reference_text_val # Use the verified text

            if not bbox:
                print(f"Warning: Missing 'bounding_box' in {json_filename} after filtering. Skipping.")
                continue
            
            # 1. Write to self_bench.txt
            bench_file.write(f"{test_index}-{text}\n")
            
            # Define output filenames
            source_filename = f"test{test_index}_source.png"
            ref_filename = f"test{test_index}_ref.png"
            mask_filename = f"test{test_index}_mask.png"
            half_ref_filename = f"test{test_index}_half_ref.png"
            half_mask_filename = f"test{test_index}_half_mask.png"
            
            # 2. Create and save the source image (just a copy)
            source_path = os.path.join(output_dir, source_filename)
            shutil.copy(image_path, source_path)
            
            # 3. Create and save the reference image (cropped from bbox)
            # bbox is [min_x, min_y, max_x, max_y]
            ref_image = original_image.crop(tuple(bbox))
            ref_path = os.path.join(output_dir, ref_filename)
            ref_image.save(ref_path)
            
            # 4. Create and save the mask image
            mask_image = Image.new('L', original_image.size, 0) # Black background
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle(tuple(bbox), fill=255) # White rectangle
            mask_path = os.path.join(output_dir, mask_filename)
            mask_image.save(mask_path)

            # --- 5. NEW: Create and save "half" versions ---
            min_x, min_y, max_x, max_y = bbox
            mid_x = min_x + (max_x - min_x) // 2
            half_bbox = (min_x, min_y, mid_x, max_y)

            # Create and save half reference image
            half_ref_image = original_image.crop(half_bbox)
            half_ref_path = os.path.join(output_dir, half_ref_filename)
            half_ref_image.save(half_ref_path)

            # Create and save half mask image
            half_mask_image = Image.new('L', original_image.size, 0)
            draw_half = ImageDraw.Draw(half_mask_image)
            draw_half.rectangle(half_bbox, fill=255)
            half_mask_path = os.path.join(output_dir, half_mask_filename)
            half_mask_image.save(half_mask_path)
            
    print("\nBenchmark dataset creation complete.")
    print(f"Dataset saved to: {output_dir}")
    print(f"Benchmark text file saved to: {bench_txt_path}")


if __name__ == '__main__':
    # The input is the output directory from the previous steps
    input_dataset_dir = config.OUTPUT_DIR 
    
    # The output is a new directory for the final benchmark set
    output_benchmark_dir = os.path.join(os.path.dirname(__file__), "Calligrapher_bench_custom")
    
    print("--- Starting Step 4: Post-processing to create benchmark dataset ---")
    create_benchmark_dataset(input_dataset_dir, output_benchmark_dir)
    print("--- Step 4 Finished ---")
