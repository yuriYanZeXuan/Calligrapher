#!/usr/bin/env python3
"""
Test script to compare custom VQAScore implementation with t2v_metrics.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Test image and text
image = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench/result_longtext_en_72.png"
text = "A visually appealing social media interface shown on a smartphone screen. At the very top, the app's name is displayed clearly as \"FitLife\", styled in bold, modern typography, colored in energizing bright orange. Underneath this title, a smaller, engaging subtitle reads, \"Empower yourself, every step counts\". At the main section of the screen, there's a featured user post with prominent textual content reading, \"Completed my first 10K run today!\" Beneath this highlighted achievement, a supportive and motivational description appears in smaller but easily readable text, \"Feeling accomplished and motivated to keep pushing myself further. Huge thanks to the FitLife community for endless inspiration!\" Just below this user-generated content, there are clearly visible social interaction buttons labeled \"Like\", \"Comment\", and \"Share\" in neat, neatly aligned blocks, each paired with subtle, simplistic icons. Towards the bottom of the screen, a neatly organized navigation bar with concise labels like \"Home\", \"Explore\", \"Progress\", and \"Profile\" appears clearly legible and easily accessible, adding functionality and encouraging seamless exploration of the content. The interface utilizes warm colors and sleek fonts, fostering an inviting and uplifting online community environment."

print("=" * 60)
print("Testing Custom VQAScore Implementation")
print("=" * 60)

# Test custom implementation
from TextCrafter_Eval.vqascore import VQAScore

print("\n[1] Loading custom VQAScore...")
custom_vqa = VQAScore(model='clip-flant5-xxl', device='cuda')

# Debug: Check model structure
print("\n[DEBUG] Checking model structure...")
print(f"  - Has vision_tower: {hasattr(custom_vqa.model, 'vision_tower')}")
print(f"  - Has mm_projector: {hasattr(custom_vqa.model, 'mm_projector')}")

if hasattr(custom_vqa.model, 'vision_tower'):
    vt = custom_vqa.model.get_vision_tower()
    print(f"  - Vision tower type: {type(vt)}")
    print(f"  - Vision tower is_loaded: {vt.is_loaded}")
    print(f"  - Vision tower name: {vt.vision_tower_name}")
    
    if hasattr(vt, 'vision_tower'):
        first_param = next(vt.vision_tower.parameters())
        print(f"  - Vision tower device: {first_param.device}")
        print(f"  - Vision tower dtype: {first_param.dtype}")
        print(f"  - Vision tower first param mean: {first_param.mean().item():.6f}")
        print(f"  - Vision tower first param std: {first_param.std().item():.6f}")

if hasattr(custom_vqa.model, 'mm_projector'):
    mp = custom_vqa.model.mm_projector
    print(f"  - mm_projector type: {type(mp)}")
    first_param = next(mp.parameters())
    print(f"  - mm_projector first param mean: {first_param.mean().item():.6f}")
    print(f"  - mm_projector first param std: {first_param.std().item():.6f}")

# Debug: Check config
print("\n[DEBUG] Checking config...")
config = custom_vqa.model.config
print(f"  - mm_vision_tower: {getattr(config, 'mm_vision_tower', 'NOT SET')}")
print(f"  - mm_projector_type: {getattr(config, 'mm_projector_type', 'NOT SET')}")
print(f"  - mm_hidden_size: {getattr(config, 'mm_hidden_size', 'NOT SET')}")
print(f"  - hidden_size: {getattr(config, 'hidden_size', 'NOT SET')}")

print("\n[2] Computing score with custom implementation...")
custom_score = custom_vqa(images=[image], texts=[text])
print(f"Custom VQAScore: {custom_score}")
print(f"Custom VQAScore (float): {float(custom_score.cpu().numpy().mean()):.6f}")

print("\n" + "=" * 60)
print("Testing t2v_metrics VQAScore (Official)")
print("=" * 60)

# Test official implementation
import t2v_metrics

print("\n[3] Loading t2v_metrics VQAScore...")
official_vqa = t2v_metrics.VQAScore(model='clip-flant5-xxl')

# Debug: Check official model structure
print("\n[DEBUG] Checking official model structure...")
official_model = official_vqa.model
print(f"  - Has vision_tower: {hasattr(official_model, 'vision_tower')}")
print(f"  - Has mm_projector: {hasattr(official_model, 'mm_projector')}")

if hasattr(official_model, 'vision_tower'):
    vt = official_model.get_vision_tower()
    print(f"  - Vision tower is_loaded: {vt.is_loaded}")
    
    if hasattr(vt, 'vision_tower'):
        first_param = next(vt.vision_tower.parameters())
        print(f"  - Vision tower first param mean: {first_param.mean().item():.6f}")
        print(f"  - Vision tower first param std: {first_param.std().item():.6f}")

if hasattr(official_model, 'mm_projector'):
    mp = official_model.mm_projector
    first_param = next(mp.parameters())
    print(f"  - mm_projector first param mean: {first_param.mean().item():.6f}")
    print(f"  - mm_projector first param std: {first_param.std().item():.6f}")

print("\n[4] Computing score with official implementation...")
official_score = official_vqa(images=[image], texts=[text])
print(f"Official VQAScore: {official_score}")
print(f"Official VQAScore (float): {float(official_score.cpu().numpy().mean()):.6f}")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
custom_val = float(custom_score.cpu().numpy().mean())
official_val = float(official_score.cpu().numpy().mean())
diff = abs(custom_val - official_val)
print(f"Custom:   {custom_val:.6f}")
print(f"Official: {official_val:.6f}")
print(f"Diff:     {diff:.6f}")
print(f"Match:    {'YES' if diff < 0.01 else 'NO'}")
