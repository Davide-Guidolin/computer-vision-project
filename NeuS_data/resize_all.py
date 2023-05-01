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
parser.add_argument('--scale', default=0.5, type=float)
parser.add_argument('--reverse', action='store_true', default=False)

args = parser.parse_args()

folder = args.folder
scale = args.scale

if args.reverse:
    scale = 1/scale

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
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    h, w, c = img.shape

    new_size = (int(w*scale), int(h*scale))

    resized_img = cv2.resize(img, new_size)

    cv2.imwrite(im_path, resized_img)


masks = os.listdir(mask_folder)
print(f"Processing {len(masks)} masks")
for i, im in enumerate(masks):
    print(f"{i}/{len(masks)}\r")
    mask_path = os.path.join(mask_folder, im)
    mask = cv2.imread(mask_path)

    h, w, c = mask.shape

    new_size = (int(w*scale), int(h*scale))

    resized_mask = cv2.resize(mask, new_size)

    cv2.imwrite(mask_path, resized_mask)


print("Done")