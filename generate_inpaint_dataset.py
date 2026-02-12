#!/usr/bin/env python3
"""
使用 Calligrapher Pass1 生成背景和布局，为 AnyText/TextFlux 创建 inpainting 数据集。

流程：
1. 加载 UnseenWords 数据集
2. 使用 Calligrapher Pass1 生成参考图和 VLM 布局规划
3. 从布局中提取 bbox 并创建 mask（矩形并集）
4. 使用 clean prompt 生成背景图（可选：直接从参考图创建）
5. 调用 AnyText 或 TextFlux 进行 inpainting
6. 保存结果到 benchmark 格式

Usage:
    python generate_inpaint_dataset.py --model anytext --output_dir ./results
    python generate_inpaint_dataset.py --model textflux --output_dir ./results
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# Setup paths
BASE_DIR = Path(__file__).parent
BASELINES_DIR = BASE_DIR / "baselines"
EVAL_DIR = BASE_DIR / "eval"

sys.path.extend([
    str(BASE_DIR),
    str(BASELINES_DIR),
    str(BASELINES_DIR / "anytext"),
    str(BASELINES_DIR / "textflux"),
])

# Model paths configuration
MODEL_PATHS = {
    'anytext': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/anytext2',
    'textflux': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill',
    'zimage': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image',
}


def load_unseenwords_dataset(data_dir: Path = None):
    """Load UnseenWords dataset."""
    if data_dir is None:
        data_dir = EVAL_DIR / "UnseenWords"
    
    data = []
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        print(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data.append({
                    'prompt': item['prompt'],
                    'text': item.get('text', []),
                    'category': item.get('category', ''),
                    'length': item.get('length', ''),
                    'text_length': item.get('text_length', 0),
                    'prompt_id': item.get('prompt_id', 0),
                    'id': f"{jsonl_file.stem}_{item.get('prompt_id', len(data))}",
                })
    
    print(f"Loaded {len(data)} items from UnseenWords")
    return data


def create_mask_from_bboxes(image_size, text_regions, expand_ratio=0.05):
    """
    从 text_regions 的 bbox 创建矩形 mask 并集。
    
    Args:
        image_size: (width, height)
        text_regions: list of dict with 'bbox' key [x_min, y_min, x_max, y_max] in normalized coords
        expand_ratio: expand bbox by this ratio for better coverage
    
    Returns:
        PIL Image (L mode, 255=mask)
    """
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for region in text_regions:
        bbox = region.get('bbox', [0, 0, 1, 1])
        x1, y1, x2, y2 = bbox
        
        # Convert normalized to pixel coords
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        
        # Expand bbox slightly
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - w * expand_ratio)
        y1 = max(0, y1 - h * expand_ratio)
        x2 = min(width, x2 + w * expand_ratio)
        y2 = min(height, y2 + h * expand_ratio)
        
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return mask


def dilate_mask(mask_img: Image.Image, kernel_size=5):
    """Dilate mask to ensure text region coverage."""
    import cv2
    mask_np = np.array(mask_img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    return Image.fromarray(dilated)


class Pass1LayoutGenerator:
    """使用 Calligrapher Pass1 生成布局和参考图。"""
    
    def __init__(self, model_path=None, device="cuda"):
        from zimage_inference import ZImageInference, GenerationConfig
        
        self.model_path = model_path or MODEL_PATHS['zimage']
        self.device = device
        
        print(f"Initializing Pass1 generator from {self.model_path}...")
        self.inference = ZImageInference(model_path=self.model_path, device=device)
        self.gen_config = GenerationConfig(
            height=1024,
            width=1024,
            num_inference_steps=20,
            seed=42,
            use_prompt_refiner=False,  # Skip refiner for speed
            use_glyph_injection=False,  # Only need Pass1
            use_harmonization=False,
        )
        self.default_seed = 42
        print("Pass1 generator initialized.")
    
    def generate_layout(self, prompt: str, text_contents: list, seed: int = None):
        """
        Generate layout using Pass1 + VLM.
        
        Returns:
            reference_image: PIL Image from Pass1
            typography_plan: dict with text_regions
            clean_prompt: str without text descriptions
        """
        from zimage_inference import GenerationConfig
        
        seed = seed or self.default_seed
        
        # Run Pass1 to get reference image
        config = GenerationConfig(
            height=1024,
            width=1024,
            num_inference_steps=20,
            seed=seed,
            use_prompt_refiner=False,
            use_glyph_injection=False,
            use_harmonization=False,
        )
        
        # Pass1: Reference image
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = self._prepare_noise(config, generator)
        
        self.inference.pipeline.scheduler.set_timesteps(
            config.num_inference_steps, device=self.device
        )
        timesteps = self.inference.pipeline.scheduler.timesteps
        
        reference_image = self.inference._run_pass1_reference(
            prompt, noise.clone(), timesteps, config
        )
        
        # VLM layout planning
        typography_plan = self.inference.vlm_agent.analyze_typography(
            reference_image, prompt, text_contents
        )
        
        # Clean prompt for background
        clean_prompt = self.inference.vlm_agent.generate_clean_prompt(prompt)
        
        return reference_image, typography_plan, clean_prompt
    
    def _prepare_noise(self, config, generator):
        """Prepare noise tensor."""
        latent_height = 2 * (config.height // (self.inference.pipeline.vae_scale_factor * 2))
        latent_width = 2 * (config.width // (self.inference.pipeline.vae_scale_factor * 2))
        num_channels = self.inference.pipeline.transformer.in_channels
        return torch.randn(
            (1, num_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
    
    def generate_background(self, clean_prompt: str):
        """Generate clean background without text."""
        from zimage_inference import GenerationConfig
        
        config = GenerationConfig(
            height=1024,
            width=1024,
            num_inference_steps=20,
            seed=42,
            use_prompt_refiner=False,
            use_glyph_injection=False,
            use_harmonization=False,
        )
        
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        
        image = self.inference.pipeline(
            prompt=clean_prompt,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        ).images[0]
        
        return image


class AnyTextInpaintWrapper:
    """Wrapper for AnyText inpainting."""
    
    def __init__(self, model_path=None):
        from inference_anytext import AnyTextInpainter
        
        self.model_path = model_path or MODEL_PATHS['anytext']
        print(f"Initializing AnyText from {self.model_path}...")
        self.inpainter = AnyTextInpainter(model_dir=self.model_path, use_fp16=True)
        print("AnyText initialized.")
    
    def generate(self, prompt: str, text_list: list, background_img: Image.Image, 
                 mask_img: Image.Image, output_path: str, seed=42):
        """Inpaint text onto background using mask."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert PIL to numpy
        source_img = np.array(background_img.convert("RGB"))
        mask_np = np.array(mask_img.convert("L"))
        
        # Format text prompt
        text_prompt = " ".join([f'"{t}"' for t in text_list])
        
        # Inpaint
        results = self.inpainter.inpaint(
            img_prompt=prompt,
            text_prompt=text_prompt,
            source_img=source_img,
            mask_img=mask_np,
            seed=seed,
            img_count=1,
            ddim_steps=20,
            strength=1.0,
            cfg_scale=7.5,
        )
        
        if results:
            img = results[0] if isinstance(results[0], Image.Image) else Image.fromarray(results[0])
            img.save(output_path)
            return True
        return False


class TextFluxInpaintWrapper:
    """Wrapper for TextFlux inpainting."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATHS['textflux']
        print(f"Initializing TextFlux from {self.model_path}...")
        
        # Import here to avoid loading if not used
        from inference_textflux import TextFluxGenerator
        self.generator = TextFluxGenerator(
            pipeline_path=self.model_path,
            transformer_path=os.path.join(self.model_path, "transformer"),
            lora_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textflux-lora-beta",
            device="cuda"
        )
        print("TextFlux initialized.")
    
    def generate(self, prompt: str, text_list: list, background_img: Image.Image,
                 mask_img: Image.Image, output_path: str, seed=42):
        """Inpaint text onto background using mask with TextFlux-specific prompt format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # TextFlux uses first text as the word to render
        text_content = text_list[0] if text_list else prompt
        
        # Resize images to 32-multiple (required by TextFlux/FluxFill)
        width, height = background_img.size
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        
        if (width, height) != (new_width, new_height):
            background_img = background_img.resize((new_width, new_height), Image.LANCZOS)
            mask_img = mask_img.resize((new_width, new_height), Image.LANCZOS)
        
        # TextFlux-specific prompt format with IMAGE1 and IMAGE2 markers
        words_str = f"'{text_content}'"
        prompt_1 = (
            "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
            "[IMAGE1] is a template image rendering the text, with the words; "
            "[IMAGE2] shows the text content naturally and correspondingly integrated into the image."
        )
        prompt_2 = (
            "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
            f"[IMAGE1] is a template image rendering the text, with the words {words_str}; "
            f"[IMAGE2] shows the text content {words_str} naturally and correspondingly integrated into the image."
        )
        
        print(f"TextFlux prompt_1: {prompt_1[:60]}...")
        print(f"TextFlux prompt_2: {prompt_2[:80]}...")
        
        # Run TextFlux inference directly with pipe to use correct prompt format
        import torch
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        result = self.generator.pipe(
            height=new_height,
            width=new_width,
            image=background_img,
            mask_image=mask_img,
            num_inference_steps=50,
            generator=generator,
            guidance_scale=30.0,  # TextFlux requires high guidance scale
            prompt=prompt_1,
            prompt_2=prompt_2,
        ).images[0]
        
        result.save(output_path)
        print(f"TextFlux result saved to {output_path}")
        return os.path.exists(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate inpainting dataset using Calligrapher Pass1 layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate using AnyText
  python generate_inpaint_dataset.py --model anytext --output_dir ./results_anytext
  
  # Generate using TextFlux
  python generate_inpaint_dataset.py --model textflux --output_dir ./results_textflux
  
  # Test with limited samples
  python generate_inpaint_dataset.py --model anytext --output_dir ./test --limit 5
  
  # Use custom UnseenWords data
  python generate_inpaint_dataset.py --model anytext --output_dir ./results \\
      --data_dir /path/to/custom/UnseenWords
        """
    )
    parser.add_argument("--model", type=str, required=True, choices=['anytext', 'textflux'],
                       help="Inpainting model to use")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="UnseenWords data directory (default: eval/UnseenWords)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--skip_pass1", action='store_true',
                       help="Skip Pass1 generation, use existing data")
    parser.add_argument("--expand_mask_ratio", type=float, default=0.05,
                       help="Expand mask bbox by this ratio")
    parser.add_argument("--dilate_mask", type=int, default=5,
                       help="Dilate mask kernel size (0=disabled)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for generation")
    args = parser.parse_args()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "layouts").mkdir(exist_ok=True)
    
    # Load dataset
    data_dir = Path(args.data_dir) if args.data_dir else None
    dataset = load_unseenwords_dataset(data_dir)
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limited to {args.limit} samples")
    
    # Initialize models
    print("\n=== Initializing Models ===")
    pass1_gen = Pass1LayoutGenerator()
    
    if args.model == 'anytext':
        inpaint_model = AnyTextInpaintWrapper()
    elif args.model == 'textflux':
        inpaint_model = TextFluxInpaintWrapper()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Process each sample
    print(f"\n=== Generating {len(dataset)} Samples ===")
    results_path = output_dir / "results.json"
    results = []
    
    # Load existing results if any
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_path}")
    
    for item in tqdm(dataset, desc="Processing"):
        item_id = item['id']
        prompt = item['prompt']
        text_list = item['text']
        
        try:
            # Step 1: Pass1 + VLM Layout
            ref_img, typography_plan, clean_prompt = pass1_gen.generate_layout(
                prompt, text_list, seed=args.seed
            )
            
            # Save layout info
            layout_dir = output_dir / "layouts"
            layout_dir.mkdir(parents=True, exist_ok=True)
            layout_path = layout_dir / f"{item_id}_layout.json"
            with open(layout_path, 'w') as f:
                json.dump({
                    'prompt': prompt,
                    'clean_prompt': clean_prompt,
                    'text': text_list,
                    'typography_plan': typography_plan,
                }, f, indent=2)
            
            # Save reference image
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            ref_path = images_dir / f"{item_id}_reference.png"
            ref_img.save(ref_path)
            
            # Step 2: Create mask from bboxes (text regions to be edited)
            text_regions = typography_plan.get('text_regions', [])
            mask_img = create_mask_from_bboxes(
                (1024, 1024), text_regions, expand_ratio=args.expand_mask_ratio
            )
            
            if args.dilate_mask > 0:
                mask_img = dilate_mask(mask_img, kernel_size=args.dilate_mask)
            
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_path = masks_dir / f"{item_id}_mask.png"
            mask_img.save(mask_path)
            
            # Step 3: Inpaint on reference image
            result_path = output_dir / "images" / f"result_{item_id}.png"
            success = inpaint_model.generate(
                prompt=prompt,
                text_list=text_list,
                background_img=ref_img,
                mask_img=mask_img,
                output_path=str(result_path),
                seed=args.seed
            )
            
            if success:
                result_item = {
                    'id': item_id,
                    'prompt': prompt,
                    'text': text_list,
                    'output': str(result_path),
                    'layout': str(layout_path),
                    'mask': str(mask_path),
                }
                results.append(result_item)
                
                # Immediately append to JSON file
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\nError processing {item_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== Done ===")
    print(f"Generated {len(results)}/{len(dataset)} samples")
    print(f"Results saved to {output_dir}")
    
    # Print evaluation command
    print(f"\nTo evaluate:")
    print(f"  python -m eval.eval_ocr --input_dir {output_dir}/images")


if __name__ == "__main__":
    main()
