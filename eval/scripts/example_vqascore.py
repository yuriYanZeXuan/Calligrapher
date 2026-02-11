#!/usr/bin/env python3
"""
Example usage of local VQAScore implementation.

This demonstrates how to use the minimal VQAScore implementation
without requiring the full t2v_metrics package.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Method 1: Direct import from TextCrafter_Eval
from eval.TextCrafter_Eval.vqascore import VQAScore

# Initialize (will download models on first run)
print("Initializing VQAScore...")
score_model = VQAScore(model='clip-flant5-xxl', device='cuda')

# Example usage (commented out - requires actual images)
# images = ["path/to/image1.png", "path/to/image2.png"]
# texts = ["A photo of a cat", "A photo of a dog"]
# scores = score_model(images, texts)
# print(f"Scores: {scores}")

# Method 2: Using through metrics module
from eval.core.metrics import VQAScoreMetrics

vqa_metrics = VQAScoreMetrics(model='clip-flant5-xxl', device='cuda')
# score = vqa_metrics.compute_score("path/to/image.png", "A photo of a cat")

print("VQAScore initialized successfully!")
print("Usage:")
print("  from eval.TextCrafter_Eval.vqascore import VQAScore")
print("  model = VQAScore(model='clip-flant5-xxl')") 
print("  scores = model(images=['image.png'], texts=['A photo'])")
