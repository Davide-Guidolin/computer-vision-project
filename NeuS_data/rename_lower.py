import argparse
import os
import sys

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
for i, name in enumerate(images):
    print(f"{i}/{len(images)}\r")
    im_path = os.path.join(image_folder, name)
    new_name = name.lower()

    new_path = os.path.join(image_folder, new_name)

    os.rename(im_path, new_path)

masks = os.listdir(mask_folder)

print(f"Processing {len(masks)} masks")
for i, name in enumerate(masks):
    print(f"{i}/{len(masks)}\r")
    im_path = os.path.join(mask_folder, name)
    new_name = name.lower()

    new_path = os.path.join(mask_folder, new_name)

    os.rename(im_path, new_path)

print("Done")