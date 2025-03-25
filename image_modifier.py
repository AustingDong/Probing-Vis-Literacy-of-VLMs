from PIL import Image
import os

def augment(img_path):
    ROOT = "./images/VLAT"
    NEW_ROOT = "./changed"
    img = Image.open(f"{ROOT}/{img_path}").convert("RGBA")

    # Create a white background
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])  # Paste using alpha channel as mask

    # Save the result
    white_bg.save(f'{NEW_ROOT}/{img_path}')  # Or save as .png if you prefer

if __name__ == "__main__":
    ROOT = "./images/VLAT"
    for img in os.listdir(ROOT):
        augment(img)