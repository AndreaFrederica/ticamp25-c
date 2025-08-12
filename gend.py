from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os, random

# Parameters
digits = "0123456789"
img_size = 64        # image dimensions (64x64)
font_size = 48       # font size for rendering (adjust to fit img_size)
output_dir = "dig" 
os.makedirs(output_dir, exist_ok=True)

# Load Times New Roman font (provide path to .ttf if needed)
font_path = "Times New Roman.ttf"  # path to Times New Roman TTF file
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()  # fallback to default font if TNR unavailable

for digit in digits:
    for i in range(200):  # e.g., 100 samples per digit
        # Create white background image
        img = Image.new('L', (img_size, img_size), color=255)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(img)
        # Determine text placement (centered)
        bbox = draw.textbbox((0, 0), digit, font=font)
        # bbox 是 (left, top, right, bottom)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((img_size-w)/2, (img_size-h)/2), digit, font=font, fill=0)
        # Apply random rotation
        angle = random.uniform(-180, 180)  
        rotated = img.rotate(angle, fillcolor=255)  # rotate around center, fill corners white
        # Optional: add slight noise
        arr = np.array(rotated, dtype=np.uint8)
        noise = np.random.normal(0, 10, arr.shape)   # Gaussian noise with sigma=10
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        rotated = Image.fromarray(arr)
        # Save image with label in filename
        filename = f"{digit}_{i}.png"
        rotated.save(os.path.join(output_dir, filename))
