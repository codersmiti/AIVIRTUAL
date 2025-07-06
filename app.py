import streamlit as st
import os
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

    zip_path = "AI_Virtual_Wardrobe/checkpoints/ACGPN_checkpoints.zip"

# Download only if not already present
    if not os.path.exists(zip_path):
        gdown.download(
            "https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx",
            zip_path,
            quiet=False
        )

# Always unzip with overwrite (no prompt)
    os.system(f'unzip -o -q "{zip_path}" -d AI_Virtual_Wardrobe/checkpoints/')

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

    # Ensure folders exist
        os.makedirs("AI_Virtual_Wardrobe/inputs/img", exist_ok=True)
        os.makedirs("AI_Virtual_Wardrobe/inputs/cloth", exist_ok=True)

        # Save inputs with expected names
        person_path = "AI_Virtual_Wardrobe/inputs/img/000001_0.png"
        cloth_path = "AI_Virtual_Wardrobe/inputs/cloth/000001_1.png"

        with open(person_path, "wb") as f:
            f.write(uploaded_img.read())
        with open(cloth_path, "wb") as f:
            f.write(uploaded_cloth.read())

        # Run core pipeline
        result = os.system("python run_pipeline.py")



    tryon_path = "C:/Users/Smiti/Downloads/AIVirtual/results/test/try-on/test_label/000001_0.png"


    if os.path.exists(tryon_path):
        st.image(tryon_path, caption="👗 Try-On Result")
    else:
        st.error("Something went wrong. Try checking input format.")


