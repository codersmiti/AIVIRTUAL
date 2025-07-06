import os
import sys
import subprocess

# Force reinstall of Pillow in case PIL is not recognized (Streamlit Cloud fix)
try:
    from PIL import Image
except ImportError:
    subprocess.run(["pip", "install", "--force-reinstall", "Pillow==10.3.0"])
    from PIL import Image

import gdown

# Add U2Net path
sys.path.append("u2net")
import u2net_load, u2net_run

# Add main app path
sys.path.append("AI_Virtual_Wardrobe")
from predict_pose import generate_pose_keypoints

# Download pose_iter_440000.caffemodel if not already present
pose_model_path = "AI_Virtual_Wardrobe/pose/pose_iter_440000.caffemodel"
if not os.path.exists(pose_model_path):
    print("Downloading pose_iter_440000.caffemodel...")
    gdown.download(id="1bcsYvDtZMdF-P8rLSmr8qQ-NvjR-4Fd5", output=pose_model_path, quiet=False)

# Image names (fixed)
img_name = "000001_0.png"
cloth_name = "000001_1.png"

# Resize and save uploaded images (already saved by Streamlit)
img_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"

img = Image.open(img_path).resize((192, 256), Image.BICUBIC)
cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")

# Save to data preprocessing folders
img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")

# Run U2Net for cloth edge extraction
u2net = u2net_load.model("u2netp")
u2net_run.infer(
    u2net,
    "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
    "AI_Virtual_Wardrobe/Data_preprocessing/test_edge"
)

# Run human parsing
os.system("python Parsing-/simple_extractor.py "
          "--dataset lip "
          "--model-restore AI_Virtual_Wardrobe/lip_final.pth "
          "--input-dir AI_Virtual_Wardrobe/Data_preprocessing/test_img "
          "--output-dir AI_Virtual_Wardrobe/Data_preprocessing/test_label")

# Run pose estimation
generate_pose_keypoints(
    f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}",
    f"AI_Virtual_Wardrobe/Data_preprocessing/test_pose/{img_name.replace('.png', '_keypoints.json')}"
)

# Create test_pairs.txt
with open("AI_Virtual_Wardrobe/Data_preprocessing/test_pairs.txt", "w") as f:
    f.write(f"{img_name} {cloth_name}")

# Run virtual try-on model
os.system("python AI_Virtual_Wardrobe/test.py")


