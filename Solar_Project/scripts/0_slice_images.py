# scripts/0_slice_images.py
import os
from patchify import patchify
import cv2
import numpy as np

# CONFIG
SOURCE_IMG_DIR = '../data/training_buildings/images/'
SOURCE_MASK_DIR = '../data/training_buildings/masks/'
OUT_IMG_DIR = '../data/training_sliced/images/'
OUT_MASK_DIR = '../data/training_sliced/masks/'
PATCH_SIZE = 512

def slice_data():
    if not os.path.exists(OUT_IMG_DIR): os.makedirs(OUT_IMG_DIR)
    if not os.path.exists(OUT_MASK_DIR): os.makedirs(OUT_MASK_DIR)

    images = os.listdir(SOURCE_IMG_DIR)
    
    for img_name in images:
        if not img_name.endswith('.tif'): continue
        
        # Load Image & Mask
        img = cv2.imread(SOURCE_IMG_DIR + img_name)
        mask = cv2.imread(SOURCE_MASK_DIR + img_name, 0) # Grayscale
        
        # Patchify (Slice into squares)
        # Note: We drop the last chunk if it doesn't fit perfectly
        patches_img = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
        patches_mask = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)
        
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                
                single_patch_img = patches_img[i, j, 0]
                single_patch_mask = patches_mask[i, j]
                
                # Save only if the mask has buildings (Optimization)
                # if np.sum(single_patch_mask) > 0: 
                cv2.imwrite(f'{OUT_IMG_DIR}{img_name}_{i}_{j}.png', single_patch_img)
                cv2.imwrite(f'{OUT_MASK_DIR}{img_name}_{i}_{j}.png', single_patch_mask)

    print("✅ Slicing Complete! Images ready for training.")

if __name__ == "__main__":
    slice_data()
