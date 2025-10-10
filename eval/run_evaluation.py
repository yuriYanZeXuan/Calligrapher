import os
import argparse
import pandas as pd
from tqdm import tqdm
import sys
import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval_ocr import OCREvaluator
from eval_dino import DinoV2Evaluator
from eval_vlm import VLMEvaluator
from eval_fid import FIDEvaluator
from utils import load_images_for_evaluation, parse_generated_filename


def main(args):
    # --- Initialize Evaluators ---
    print("Initializing all evaluators...")
    ocr_evaluator = OCREvaluator()
    dino_evaluator = DinoV2Evaluator(device=args.device)
    vlm_evaluator = None
    if args.run_vlm:
        try:
            vlm_evaluator = VLMEvaluator()
        except ValueError as e:
            print(f"Could not initialize VLM Evaluator: {e}")
            print("Skipping VLM evaluations.")
            args.run_vlm = False
    
    # --- Find Generated Images ---
    generated_images = [os.path.join(args.generated_dir, f) for f in os.listdir(args.generated_dir) if f.startswith('result_') and f.endswith('.png')]
    if not generated_images:
        print(f"Error: No generated images found in {args.generated_dir}")
        return

    # --- Run Per-Image Evaluations ---
    results = []
    for img_path in tqdm(generated_images, desc="Evaluating Images"):
        try:
            gen_img, _, mask_img, ref_img, metadata = load_images_for_evaluation(img_path, args.benchmark_dir)
            if gen_img is None:
                continue
        except Exception as e:
            print(f"Skipping {os.path.basename(img_path)} due to loading error: {e}")
            continue

        result_row = {
            'filename': os.path.basename(img_path),
            'prompt': metadata['prompt'],
            'ref_id': metadata['ref_id'],
        }

        # OCR Accuracy
        result_row['ocr_accuracy'] = ocr_evaluator.calculate_ocr_accuracy(gen_img, metadata['prompt'])
        
        # DINO Similarity
        result_row['dino_similarity'] = dino_evaluator.calculate_similarity(gen_img, ref_img, mask_img)
        
        # VLM Scores
        if args.run_vlm and vlm_evaluator:
            aesthetic_score, _ = vlm_evaluator.evaluate_aesthetic(gen_img, ref_img)
            match_score, _ = vlm_evaluator.evaluate_text_image_match(gen_img, metadata['prompt'])
            result_row['vlm_aesthetic_score'] = aesthetic_score
            result_row['vlm_text_match_score'] = match_score
        else:
            result_row['vlm_aesthetic_score'] = None
            result_row['vlm_text_match_score'] = None

        results.append(result_row)
        
        # Save intermediate results
        df_intermediate = pd.DataFrame(results)
        df_intermediate.to_csv(args.output_csv, index=False)


    # --- Run Distribution-Based Evaluation (FID) ---
    fid_score = None
    if args.run_fid:
        print("\nCalculating Masked FID score (this may take a while)...")
        fid_evaluator = FIDEvaluator(device=args.device)
        # The "real" images for FID are the source images from the benchmark
        real_images_dir = args.benchmark_dir
        mask_dir = args.benchmark_dir
        fid_score = fid_evaluator.evaluate(args.generated_dir, real_images_dir, mask_dir)
        print(f"Masked FID Score: {fid_score}")

    # --- Final Report ---
    df = pd.DataFrame(results)
    
    print("\n--- Evaluation Summary ---")
    # Calculate and print mean scores for all numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    mean_scores = df[numeric_cols].mean()
    print(mean_scores)
    
    if fid_score is not None:
        print(f"\nMasked FID Score: {fid_score:.4f} (Lower is better)")
        # Add FID score as a new column to the dataframe with the value in all rows
        df['masked_fid_score'] = fid_score
        
    print(f"\nDetailed results saved to: {args.output_csv}")
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full evaluation suite for Calligrapher.")
    parser.add_argument("--generated_dir", type=str, required=False, default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/cli_exps/2025-10-10-17-12-59_self",
                        help="Directory containing the generated images from inference.")
    parser.add_argument("--benchmark_dir", type=str, required=False, default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing",
                        help="Directory of the benchmark dataset (e.g., Calligrapher_bench_testing).")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv",
                        help="Path to save the detailed CSV results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run torch models on ('cuda' or 'cpu').")
    parser.add_argument("--no-vlm", action="store_false", dest="run_vlm",
                        help="Skip the VLM-based evaluations (Aesthetic and Text-Match).")
    parser.add_argument("--no-fid", action="store_false", dest="run_fid",
                        help="Skip the FID evaluation.")
    
    args = parser.parse_args()
    
    # Ensure API_KEY is set if VLM is to be run
    if args.run_vlm and not os.environ.get("API_KEY"):
        print("="*50)
        print("!!! WARNING: API_KEY is not set. VLM evaluations will be skipped. !!!")
        print("Please set it to your OpenAI API key to run VLM evaluation.")
        print("Example: export API_KEY='sk-...'")
        print("="*50)
        args.run_vlm = False

    main(args)
