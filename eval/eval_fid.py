import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
import os
import sys
import re

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import parse_generated_filename


class MaskedImageDataset(Dataset):
    def __init__(self, image_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Determine corresponding mask path
        metadata = parse_generated_filename(img_path)
        if not metadata:
            # Fallback for real images (e.g., test1_source.png)
            # We need to extract the 'test_id' part (e.g., 'test1')
            base_name = os.path.basename(img_path)
            match = re.match(r'(test\d+)_', base_name)
            if match:
                ref_id = match.group(1)
                mask_path = os.path.join(self.mask_dir, f"{ref_id}_mask.png")
            else:
                # If it doesn't match, we can't find the mask. Set to None.
                mask_path = None
        else:
            mask_path = os.path.join(self.mask_dir, f"{metadata['ref_id']}_mask.png")

        try:
            image = Image.open(img_path).convert("RGB")
            # Handle case where mask_path could not be determined
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                print(f"Warning: Mask not found for {img_path}. Skipping masking.")
                mask = Image.new('L', image.size, 255) # White mask = no masking
        except FileNotFoundError:
            print(f"Warning: Mask file not found at {mask_path} for {img_path}. Skipping masking.")
            mask = Image.new('L', image.size, 255) # White mask = no masking

        # Apply mask: black out unmasked area
        mask = mask.resize(image.size, Image.NEAREST)
        img_np = np.array(image)
        mask_np = np.array(mask)
        img_np[mask_np == 0] = 0
        image = Image.fromarray(img_np)

        if self.transform:
            image = self.transform(image)
            
        return image


class FIDEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("Initializing InceptionV3 for FID...")
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights, transform_input=False).to(self.device)
        self.model.fc = torch.nn.Identity() # Use up to the last pooling layer
        self.model.eval()
        self.transform = weights.transforms()
        print("InceptionV3 initialized.")

    @torch.no_grad()
    def get_activations(self, dataloader):
        """Calculates the activations of the Inception network for a dataset."""
        activations = []
        for batch in dataloader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            activations.append(pred.cpu().numpy())
        return np.concatenate(activations, axis=0)

    def calculate_fid(self, act1, act2):
        """Calculates FID score based on activations."""
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # Matrix square root of the product of covariances
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # Check for imaginary numbers, which can occur if the matrix is not positive semi-definite
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def evaluate(self, generated_dir, real_dir, mask_dir, batch_size=16):
        """
        Main evaluation function.
        - generated_dir: Path to directory with generated images.
        - real_dir: Path to directory with corresponding real images (e.g., source images).
        - mask_dir: Path to directory with masks.
        """
        gen_paths = [os.path.join(generated_dir, f) for f in os.listdir(generated_dir) if f.endswith('.png')]
        real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.png')]

        # For this task, the "real" images should be the source images that correspond to the generated ones
        # We assume a 1-to-1 mapping based on ref_id, so we'll filter real_paths
        gen_ref_ids = {parse_generated_filename(p)['ref_id'] for p in gen_paths if parse_generated_filename(p)}
        real_paths_filtered = [p for p in real_paths if any(ref_id in p for ref_id in gen_ref_ids)]
        
        if not gen_paths or not real_paths_filtered:
            print("Error: No images found in generated or real directories.")
            return float('inf')

        gen_dataset = MaskedImageDataset(gen_paths, mask_dir, self.transform)
        real_dataset = MaskedImageDataset(real_paths_filtered, mask_dir, self.transform)

        gen_loader = DataLoader(gen_dataset, batch_size=batch_size)
        real_loader = DataLoader(real_dataset, batch_size=batch_size)

        act_gen = self.get_activations(gen_loader)
        act_real = self.get_activations(real_loader)
        
        fid_score = self.calculate_fid(act_gen, act_real)
        return fid_score


def main():
    """
    Example usage of the FIDEvaluator.
    NOTE: FID requires a distribution of images, so running on a single image is not meaningful.
    This example assumes you have a folder of generated images and a folder of real source images.
    """
    # --- IMPORTANT ---
    # Change these paths to your local paths
    # This should be the folder where your `infer_calligrapher_self_custom.py` saves results
    generated_images_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/cli_exps/2025-10-10-17-12-59_self" 
    # This is the original benchmark dataset folder
    benchmark_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
    # -----------------
    
    # For FID, the "real" images are the source images, and masks are in the same dir
    real_images_dir = benchmark_dir
    mask_dir = benchmark_dir
    
    if not os.path.exists(generated_images_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `eval_fid.py` main function !!!")
        print(f"Path checked: {generated_images_dir}")
        print("="*50)
        return
        
    print(f"Calculating Masked FID score...")
    print(f"Generated Images: {generated_images_dir}")
    print(f"Real Images (source): {real_images_dir}")
    print(f"Masks: {mask_dir}")
    
    evaluator = FIDEvaluator()
    fid_score = evaluator.evaluate(generated_images_dir, real_images_dir, mask_dir)
    
    print(f"\nMasked FID Score: {fid_score:.4f}")
    print("(Lower is better)")


if __name__ == '__main__':
    main()
