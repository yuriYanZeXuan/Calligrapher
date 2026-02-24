#!/usr/bin/env python3
"""
Table 1: VLM Agent Typography Planning Effectiveness (Ablation Study)

Evaluate 4 configurations:
  1. Baseline:      Fixed rule-based layout (center region, equal division)
  2. w/o Grid:      VLM planning without grid overlay
  3. w/o Agent:     Random bbox sampling
  4. Ours (Full):   VLM + 10×10 Grid Overlay

Metrics:
  - Layout Match:   IoU between predicted bbox and ground truth
  - Text Acc:       OCR accuracy (future work)

Ground truth is constructed by randomly sampling position and scale for each formula.
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infer.formula_helper import render_formula, plaintext_to_latex
from infer.VLM_agent import VLMAgent, _add_grid_overlay


@dataclass
class BBox:
    """Bounding box in normalized coordinates [x_min, y_min, x_max, y_max]"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x1 = int(self.x_min * width)
        y1 = int(self.y_min * height)
        x2 = int(self.x_max * width)
        y2 = int(self.y_max * height)
        return (x1, y1, x2, y2)
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: "BBox") -> float:
        """Compute IoU with another bbox"""
        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)
        x2 = min(self.x_max, other.x_max)
        y2 = min(self.y_max, other.y_max)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


def load_formulas(jsonl_path: str, num_samples: int = 10, seed: int = 42) -> list:
    """Load formulas from jsonl file"""
    random.seed(seed)
    formulas = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            formulas.append(data)
    
    # Random sample
    if num_samples < len(formulas):
        formulas = random.sample(formulas, num_samples)
    
    return formulas


def generate_random_bbox(
    img_width: int,
    img_height: int,
    text_aspect_ratio: float = 2.0,
    min_scale: float = 0.15,
    max_scale: float = 0.6,
    seed: Optional[int] = None,
) -> BBox:
    """
    Generate random bbox as ground truth.
    
    Args:
        text_aspect_ratio: width / height ratio of the text region
        min_scale/max_scale: min/max scale of bbox relative to image
    """
    if seed is not None:
        random.seed(seed)
    
    # Random scale
    scale = random.uniform(min_scale, max_scale)
    
    # Calculate size maintaining aspect ratio
    # We want bbox_height * bbox_width = scale * img_area
    # And bbox_width / bbox_height = text_aspect_ratio
    img_area = img_width * img_height
    bbox_area = scale * scale * img_area  # scale is relative to each dimension
    
    bbox_height = (bbox_area / text_aspect_ratio) ** 0.5
    bbox_width = bbox_height * text_aspect_ratio
    
    # Normalize
    h_norm = bbox_height / img_height
    w_norm = bbox_width / img_width
    
    # Random position (ensure within bounds)
    x_min = random.uniform(0.0, max(0.0, 1.0 - w_norm))
    y_min = random.uniform(0.0, max(0.0, 1.0 - h_norm))
    x_max = x_min + w_norm
    y_max = y_min + h_norm
    
    return BBox(x_min, y_min, x_max, y_max)


def render_formula_on_canvas(
    formula: str,
    canvas_width: int,
    canvas_height: int,
    bbox: BBox,
    text_color: str = "white",
    background_color: str = "black",
) -> Image.Image:
    """Render formula on a canvas at specified bbox location"""
    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    
    # Calculate pixel bbox
    x1, y1, x2, y2 = bbox.to_pixel(canvas_width, canvas_height)
    region_width = max(x2 - x1, 32)
    region_height = max(y2 - y1, 32)
    
    # Render formula
    formula_img = render_formula(
        formula,
        region_width,
        region_height,
        text_color=text_color,
        background_color=background_color,
        force_latex=True,
    )
    
    # Paste onto canvas
    canvas.paste(formula_img, (x1, y1))
    
    return canvas


def baseline_predict_bbox(img_width: int, img_height: int) -> BBox:
    """Baseline: Fixed center region"""
    # Fixed center region: 0.2-0.8 in both dimensions
    return BBox(0.2, 0.35, 0.8, 0.65)


def random_predict_bbox(seed: Optional[int] = None) -> BBox:
    """w/o Agent: Random bbox sampling"""
    if seed is not None:
        random.seed(seed)
    return BBox(
        random.uniform(0.05, 0.4),
        random.uniform(0.05, 0.4),
        random.uniform(0.6, 0.95),
        random.uniform(0.6, 0.95),
    )


def vlm_predict_bbox(
    image: Image.Image,
    formula: str,
    vlm_agent: VLMAgent,
    use_grid: bool = True,
    grid_size: int = 11,
) -> Optional[BBox]:
    """
    Use VLM to predict bbox.
    
    Args:
        use_grid: Whether to add grid overlay
        grid_size: Number of grid lines (4 for 3x3, 6 for 5x5, 11 for 10x10, 13 for 12x12)
    """
    # Prepare image
    if use_grid:
        image_with_prompt = _add_grid_overlay(image, grid_size=grid_size)
    else:
        image_with_prompt = image.copy()
    
    # Calculate grid cells from grid_size
    grid_cells = grid_size - 1
    
    # Construct prompt
    system_prompt = (
        "You are an expert in image typography analysis. "
        "Given an image with a formula, predict the bounding box that would best fit the formula.\n\n"
    )
    
    if use_grid:
        system_prompt += (
            f"The image has a {grid_cells}×{grid_cells} grid overlay with coordinate annotations (0.0-1.0). "
            f"Use the grid coordinates to specify the bbox precisely.\n\n"
        )
    
    system_prompt += (
        "Output the bbox in JSON format:\n"
        "{\"bbox\": [x_min, y_min, x_max, y_max]}\n\n"
        "All coordinates should be normalized (0.0-1.0). "
        "Ensure the bbox fully contains the formula with appropriate margins."
    )
    
    user_content = f'The formula to place is: "{formula}"'
    
    try:
        response = vlm_agent.call_vlm(
            "analyze_typography",  # Use existing template
            user_content,
            images=[image_with_prompt],
            max_tokens=256,
            temperature=0.3,
        )
        
        # Extract bbox from response
        bbox = extract_bbox_from_response(response)
        return bbox
    except Exception as e:
        print(f"VLM prediction failed: {e}")
        return None


def extract_bbox_from_response(response: str) -> Optional[BBox]:
    """Extract bbox from VLM response"""
    # Try JSON extraction
    try:
        # Look for JSON pattern
        match = re.search(r"\{[^}]*\"bbox\"\s*:\s*\[[^\]]+\]", response)
        if match:
            json_str = match.group(0) + "}"
            data = json.loads(json_str)
            bbox_list = data["bbox"]
            return BBox(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])
        
        # Try direct array pattern
        match = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", response)
        if match:
            nums = [float(match.group(i)) for i in range(1, 5)]
            return BBox(nums[0], nums[1], nums[2], nums[3])
    except Exception as e:
        print(f"Failed to parse bbox: {e}")
    
    return None


def visualize_result(
    image: Image.Image,
    gt_bbox: BBox,
    pred_bbox: BBox,
    iou: float,
    config_name: str,
) -> Image.Image:
    """Visualize ground truth and predicted bbox"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Draw ground truth (green)
    x1, y1, x2, y2 = gt_bbox.to_pixel(width, height)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    
    # Draw prediction (red)
    x1, y1, x2, y2 = pred_bbox.to_pixel(width, height)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Add text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    text = f"{config_name}: IoU={iou:.3f}"
    draw.text((10, 10), text, fill="yellow", font=font)
    draw.text((10, 40), "Green=GT, Red=Pred", fill="yellow", font=font)
    
    return img


def run_single_config(
    formulas: list,
    config_name: str,
    img_width: int,
    img_height: int,
    vlm_agent: Optional[VLMAgent],
    output_dir: Path,
    visualize: bool = True,
) -> dict:
    """
    Run evaluation for a single configuration.
    
    Returns metrics dict.
    """
    ious = []
    
    for idx, data in enumerate(formulas):
        formula = data["text"][0]
        
        # Generate ground truth bbox
        gt_bbox = generate_random_bbox(
            img_width, img_height,
            text_aspect_ratio=2.0,
            seed=idx + 42,
        )
        
        # Render image with formula at GT location
        image = render_formula_on_canvas(
            formula, img_width, img_height, gt_bbox,
            text_color="white", background_color="black",
        )
        
        # Predict bbox based on configuration
        if config_name == "Baseline":
            pred_bbox = baseline_predict_bbox(img_width, img_height)
        elif config_name == "w/o Agent":
            pred_bbox = random_predict_bbox(seed=idx + 100)
        elif config_name in ["w/o Grid", "Ours (10×10 Grid)"] or "Grid" in config_name:
            if vlm_agent is None:
                raise ValueError(f"VLM agent required for {config_name}")
            # Determine grid size based on config name
            if config_name == "w/o Grid":
                grid_size = 0  # Not used
                use_grid = False
            elif "3×3" in config_name or "3x3" in config_name:
                grid_size = 4  # 4 lines = 3 cells
                use_grid = True
            elif "5×5" in config_name or "5x5" in config_name:
                grid_size = 6  # 6 lines = 5 cells
                use_grid = True
            elif "8×8" in config_name or "8x8" in config_name:
                grid_size = 9  # 9 lines = 8 cells
                use_grid = True
            elif "12×12" in config_name or "12x12" in config_name:
                grid_size = 13  # 13 lines = 12 cells
                use_grid = True
            else:  # 10×10 default
                grid_size = 11  # 11 lines = 10 cells
                use_grid = True
            pred_bbox = vlm_predict_bbox(image, formula, vlm_agent, use_grid=use_grid, grid_size=grid_size)
            if pred_bbox is None:
                print(f"  Skipping {config_name} sample {idx} due to VLM failure")
                continue
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        # Compute IoU
        iou = gt_bbox.iou(pred_bbox)
        ious.append(iou)
        
        # Visualize
        if visualize:
            vis_img = visualize_result(image, gt_bbox, pred_bbox, iou, config_name)
            vis_path = output_dir / f"{config_name.replace(' ', '_').replace('/', '_')}_sample{idx:03d}.png"
            vis_img.save(vis_path)
        
        print(f"  Sample {idx}: IoU={iou:.4f}")
    
    # Compute statistics
    metrics = {
        "config": config_name,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "max_iou": float(np.max(ious)) if ious else 0.0,
        "num_samples": len(ious),
    }
    
    return metrics


def print_results_table(all_metrics: list):
    """Print results in a formatted table"""
    print("\n" + "=" * 90)
    print("TABLE 1: VLM Agent Typography Planning Effectiveness (Grid Ablation)")
    print("=" * 90)
    print(f"{'Configuration':<25} {'Mean IoU':>12} {'Median IoU':>12} {'Std':>10} {'Samples':>10}")
    print("-" * 90)
    
    for m in all_metrics:
        print(f"{m['config']:<25} {m['mean_iou']:>12.4f} {m['median_iou']:>12.4f} {m['std_iou']:>10.4f} {m['num_samples']:>10}")
    
    print("=" * 90)
    
    # Print improvement over baseline
    baseline_mean = None
    for m in all_metrics:
        if m["config"] == "Baseline":
            baseline_mean = m["mean_iou"]
            break
    
    if baseline_mean and baseline_mean > 0:
        print("\nImprovement over Baseline:")
        for m in all_metrics:
            if m["config"] != "Baseline":
                improvement = (m["mean_iou"] - baseline_mean) / baseline_mean * 100
                print(f"  {m['config']}: {improvement:+.1f}%")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Table 1: VLM Agent Typography Planning Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="/Users/yanzexuan/code/Calligrapher/eval/UnseenWords/unseen_mid_sci.jsonl",
        help="Path to formula jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/yanzexuan/code/Calligrapher/ablation/output_table1",
        help="Output directory for visualization and results",
    )
    
    # Experiment settings
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of formulas to evaluate",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # VLM settings
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen3-vl-235b-a22b-instruct",
        help="VLM model name",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM-based configs (for quick testing)",
    )
    
    # Visualization
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization output",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load formulas
    print(f"Loading formulas from {args.jsonl_path}")
    formulas = load_formulas(args.jsonl_path, args.num_samples, args.seed)
    print(f"Loaded {len(formulas)} formulas")
    
    # Initialize VLM agent
    vlm_agent = None
    if not args.skip_vlm:
        print(f"Initializing VLM agent with model: {args.vlm_model}")
        vlm_agent = VLMAgent(model=args.vlm_model)
    else:
        print("Skipping VLM initialization (--skip-vlm)")
    
    # Define configurations to evaluate
    configs = [
        "Baseline",
        "w/o Agent",
    ]
    if not args.skip_vlm:
        # Grid ablation: 3x3, 5x5, 8x8
        configs.extend([
            "w/o Grid",
            "VLM + 3×3 Grid",
            "VLM + 5×5 Grid",
            "VLM + 8×8 Grid",
        ])
    
    # Run evaluation
    all_metrics = []
    visualize = not args.no_visualize
    
    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")
        
        metrics = run_single_config(
            formulas,
            config_name,
            args.img_width,
            args.img_height,
            vlm_agent,
            output_dir,
            visualize=visualize,
        )
        
        all_metrics.append(metrics)
        print(f"\n{config_name} Results:")
        print(f"  Mean IoU:   {metrics['mean_iou']:.4f}")
        print(f"  Median IoU: {metrics['median_iou']:.4f}")
        print(f"  Std:        {metrics['std_iou']:.4f}")
    
    # Print summary table
    print_results_table(all_metrics)
    
    # Save results to JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": all_metrics,
            "config": vars(args),
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
