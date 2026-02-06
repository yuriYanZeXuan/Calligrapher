#!/usr/bin/env python3
"""Z-Image 推理启动脚本 - 支持多 GPU"""
import os
import argparse
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="爱因斯坦站在黑板前，写着二次方程求根公式")
    parser.add_argument("--text", default="x = (-b ± √(b²-4ac)) / 2a")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam", type=int, default=8, help="beam size for TTS")
    parser.add_argument("--tts", action="store_true", help="启用 Test Time Scaling")
    parser.add_argument("--gpus", type=str, default=None, help="GPU列表，如 0,1,2,3")
    args = parser.parse_args()
    
    # 解析 GPU
    if args.gpus:
        devices = [f"cuda:{i}" for i in args.gpus.split(",")]
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
    print(f"使用 GPU: {devices}")
    
    if args.tts and len(devices) > 1:
        # 多 GPU TTS
        from infer.test_time_scaling import create_multi_gpu_tts
        from infer.prompt_refiner import PromptRefiner
        
        tts = create_multi_gpu_tts(
            model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image",
            devices=devices,
            prompt_refiner=PromptRefiner()
        )
        
        image, score = tts.generate_with_beam_search(
            prompt=args.prompt,
            text_content=args.text,
            beam_size=args.beam,
            seed=args.seed
        )
        print(f"最终得分: {score:.2f}")
    else:
        # 单 GPU
        from zimage_inference import ZImageInference
        
        inference = ZImageInference(device=devices[0])
        image = inference.generate(
            prompt=args.prompt,
            text_regions=[{"bbox": [0.2, 0.1, 0.8, 0.5], "content": args.text}],
            seed=args.seed
        )
    
    image.save(args.output)
    print(f"保存到 {args.output}")


if __name__ == "__main__":
    main()
