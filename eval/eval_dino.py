import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import os
import sys

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import load_images_for_evaluation


class DinoV2Evaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("Initializing DINOv2...")
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
        self.model.eval()
        print("DINOv2 initialized.")
        
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def get_embedding(self, image: Image.Image, mask: Image.Image = None):
        """
        Get the DINOv2 embedding for an image, optionally applying a mask.
        If a mask is provided, the unmasked area will be blacked out.
        """
        if mask:
            # Resize mask to match image
            mask_resized = mask.resize(image.size, Image.NEAREST)
            img_np = np.array(image)
            mask_np = np.array(mask_resized)
            
            # Black out the unmasked area
            img_np[mask_np == 0] = 0
            image = Image.fromarray(img_np)

        # Transform and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get CLS token embedding
        embedding = self.model(image_tensor)
        return embedding

    def calculate_similarity(self, generated_img: Image.Image, ref_img: Image.Image, mask: Image.Image):
        """
        Calculates the cosine similarity between the masked generated image and the reference image.
        """
        # Get embedding for the generated image within the masked region
        generated_embedding = self.get_embedding(generated_img, mask)
        
        # Get embedding for the full reference image (to capture the style)
        ref_embedding = self.get_embedding(ref_img)
        
        similarity = cosine_similarity(generated_embedding, ref_embedding).item()
        return similarity


def main():
    """
    Example usage of the DinoV2Evaluator.
    """
    # --- IMPORTANT ---
    # Change these paths to your local paths
    generated_image_path = "/path/to/your/generated/images/result_0_test1_The_text_is_Calligrapher_12345678.png"
    benchmark_dir = "/Users/yanzexuan/code/dataset/Calligrapher_bench_testing"
    # -----------------
    
    if not os.path.exists(generated_image_path) or not os.path.exists(benchmark_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `eval_dino.py` main function !!!")
        print(f"File path checked: {generated_image_path}")
        print("="*50)
        return

    generated_img, _, mask_img, ref_img, _ = load_images_for_evaluation(generated_image_path, benchmark_dir)

    if generated_img is None:
        print("Failed to load images.")
        return

    print(f"Evaluating file: {os.path.basename(generated_image_path)}")

    evaluator = DinoV2Evaluator()
    similarity = evaluator.calculate_similarity(generated_img, ref_img, mask_img)

    print(f"\nDINOv2 Feature Similarity (masked): {similarity:.4f}")


if __name__ == '__main__':
    main()
