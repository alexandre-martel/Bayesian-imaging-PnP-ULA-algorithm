import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import numpy as np

def calculate_metrics(path1, path2):
 img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
 img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

 if img1.shape != img2.shape:
  print(f"Warning, images have different shapes.")
  print(f"Image 1: {img1.shape} | Image 2: {img2.shape}")
  
 img1 = img1.astype(np.float32) / 255.0
 img2 = img2.astype(np.float32) / 255.0

 score = psnr_func(img1, img2, data_range=1.0)
    
 ssim_val = ssim_func(img1, img2, data_range=1.0)
    
 return score, ssim_val

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Calculate PSNR between two images")
 parser.add_argument("img1", help="Path to the first image")
 parser.add_argument("img2", help="Path to the second image")
 
 args = parser.parse_args()

 result = calculate_metrics(args.img1, args.img2)
 print(f"---")
 print(f"PSNR : {result[0]:.2f} dB")
 print(f"SSIM : {result[1]:.2f}")
 print(f"---")