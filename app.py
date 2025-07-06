import streamlit as st
import os
import sys
import time
from PIL import Image
import numpy as np
import gdown

# === SETUP ===

@st.cache_resource
def setup_environment():
    # Clone repos
    os.system("git clone https://github.com/jayneel-shah18/AI_Virtual_Wardrobe.git")
    os.system("git clone https://github.com/codersmiti/Parsing-.git")
    os.system("git clone https://github.com/jayneel-shah18/u2net.git")

    # Download U2Net model
    os.makedirs("u2net/saved_models/u2netp", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
        "u2net/saved_models/u2netp/u2netp.pth",
        quiet=False
    )

    # Download ACGPN checkpoints
    os.makedirs("AI_Virtual_Wardrobe/checkpoints", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx",
        "AI_Virtual_Wardrobe/checkpoints/ACGPN_checkpoints.zip",
        quiet=False
    )
    os.system("unzip AI_Virtual_Wardrobe/checkpoints/ACGPN_checkpoints.zip -d AI_Virtual_Wardrobe/checkpoints/")

    # Download human parsing model
    gdown.download(
        "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
        "AI_Virtual_Wardrobe/lip_final.pth",
        quiet=False
    )

    # Folder structure
    subdirs = [
        "inputs/img", "inputs/cloth",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_edge",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_label",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_pose",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_mask",
        "AI_Virtual_Wardrobe/Data_preprocessing/test_colormask",
    ]
    for d in subdirs:
        os.makedirs(d, exist_ok=True)

setup_environment()

# === STREAMLIT UI ===

st.title("👗 AI Virtual Try-On")
st.markdown("Upload your image and cloth to see the virtual try-on result.")

uploaded_img = st.file_uploader("Upload your person image", type=["jpg", "png"], key="person")
uploaded_cloth = st.file_uploader("Upload your cloth image", type=["jpg", "png"], key="cloth")

if uploaded_img and uploaded_cloth:
    st.success("Images uploaded. Generating try-on...")
    with st.spinner("Processing..."):

        # Save inputs
        img_name = "000001_0.png"
        cloth_name = "000001_1.png"
        person_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
        cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"
        with open(person_path, "wb") as f:
            f.write(uploaded_img.read())
        with open(cloth_path, "wb") as f:
            f.write(uploaded_cloth.read())

        # === INFERENCES ===
        sys.path.append("u2net")
        import u2net_load, u2net_run
        sys.path.append("AI_Virtual_Wardrobe")
        from predict_pose import generate_pose_keypoints

        # Resize + Save
        img = Image.open(person_path).resize((192, 256), Image.BICUBIC)
        cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")
        img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
        cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")

        # Run U2Net edge detection
        u2net = u2net_load.model("u2netp")
        u2net_run.infer(u2net,
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_edge")

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

        # Run try-on model
        os.system("python AI_Virtual_Wardrobe/test.py")

    tryon_path = "results/test/try-on/test_label/000001_0.png"
    if os.path.exists(tryon_path):
        st.image(tryon_path, caption="👗 Try-On Result")
    else:
        st.error("Something went wrong. Try checking input format.")



