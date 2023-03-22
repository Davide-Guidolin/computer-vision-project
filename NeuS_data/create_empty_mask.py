import argparse
import os
import sys
import cv2
import numpy as np

def check_folder(folder):
    if not os.path.exists(folder):
        print(f"{folder} does not exists")
        sys.exit(1)

parser = argparse.ArgumentParser()

parser.add_argument('--folder')

args = parser.parse_args()

folder = args.folder

check_folder(folder)

image_folder = os.path.join(folder, 'image')
mask_folder = os.path.join(folder, 'mask')

check_folder(image_folder)
check_folder(mask_folder)

images = os.listdir(image_folder)

print(f"Processing {len(images)} images")
for i, im in enumerate(images):
    print(f"{i}/{len(images)}\r")
    im_path = os.path.join(image_folder, im)
    img = cv2.imread(im_path)

    h, w, c = img.shape

    white_im = np.ones([h, w, c], dtype=np.uint8) * 255

    mask_path = os.path.join(mask_folder, im)
    cv2.imwrite(mask_path, white_im)

print("Done")