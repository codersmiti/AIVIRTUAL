import streamlit as st
import os
import sys
import subprocess
from PIL import Image
import numpy as np
import gdown

# === SETUP ===
@st.cache_resource
def setup_environment():
    """Setup environment by cloning repos and downloading models"""
    try:
        # Clone repos
        if not os.path.exists("AI_Virtual_Wardrobe"):
            os.system("git clone https://github.com/jayneel-shah18/AI_Virtual_Wardrobe.git")
        
        if not os.path.exists("Parsing-"):
            os.system("git clone https://github.com/codersmiti/Parsing-.git")
        
        if not os.path.exists("u2net"):
            os.system("git clone https://github.com/jayneel-shah18/u2net.git")
        
        # Download U2Net model
        u2net_model_path = "u2net/saved_models/u2netp/u2netp.pth"
        if not os.path.exists(u2net_model_path):
            os.makedirs("u2net/saved_models/u2netp", exist_ok=True)
            gdown.download(
                "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
                u2net_model_path,
                quiet=False
            )
        
        # Download ACGPN checkpoints
        checkpoint_zip = "AI_Virtual_Wardrobe/checkpoints/ACGPN_checkpoints.zip"
        if not os.path.exists(checkpoint_zip):
            os.makedirs("AI_Virtual_Wardrobe/checkpoints", exist_ok=True)
            gdown.download(
                "https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx",
                checkpoint_zip,
                quiet=False
            )
            
            # Extract using Python zipfile instead of system unzip
            import zipfile
            with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
                zip_ref.extractall("AI_Virtual_Wardrobe/checkpoints/")
        
        # Download human parsing model
        parsing_model_path = "AI_Virtual_Wardrobe/lip_final.pth"
        if not os.path.exists(parsing_model_path):
            gdown.download(
                "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
                parsing_model_path,
                quiet=False
            )
        
        # Create folder structure
        subdirs = [
            "inputs/img", "inputs/cloth",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_edge",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_label",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_pose",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_mask",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_colormask",
            "AI_Virtual_Wardrobe/inputs/img",
            "AI_Virtual_Wardrobe/inputs/cloth",
        ]
        for d in subdirs:
            os.makedirs(d, exist_ok=True)
            
        st.success("Environment setup complete!")
        return True
        
    except Exception as e:
        st.error(f"Setup failed: {str(e)}")
        return False

def run_pipeline_function():
    """Run the entire pipeline as a function instead of separate process"""
    try:
        # Add paths to sys.path
        sys.path.append("u2net")
        sys.path.append("AI_Virtual_Wardrobe")
        
        # Import required modules
        import u2net_load, u2net_run
        from predict_pose import generate_pose_keypoints
        
        # Download pose model if needed
        pose_model_path = "AI_Virtual_Wardrobe/pose/pose_iter_440000.caffemodel"
        if not os.path.exists(pose_model_path):
            os.makedirs("AI_Virtual_Wardrobe/pose", exist_ok=True)
            gdown.download(id="1bcsYvDtZMdF-P8rLSmr8qQ-NvjR-4Fd5", output=pose_model_path, quiet=False)
        
        # Image names (fixed)
        img_name = "000001_0.png"
        cloth_name = "000001_1.png"
        
        # Image paths
        img_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
        cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"
        
        # Resize and save images
        img = Image.open(img_path).resize((192, 256), Image.BICUBIC)
        cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")
        
        # Save to data preprocessing folders
        img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
        cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")
        
        # Run U2Net for cloth edge extraction
        u2net = u2net_load.model("u2netp")
        u2net_run.infer(u2net,
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_edge")
        
        # Run human parsing
        subprocess.run([
            sys.executable, "Parsing-/simple_extractor.py",
            "--dataset", "lip",
            "--model-restore", "AI_Virtual_Wardrobe/lip_final.pth",
            "--input-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
            "--output-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_label"
        ], check=True)
        
        # Run pose estimation
        generate_pose_keypoints(
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}",
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_pose/{img_name.replace('.png', '_keypoints.json')}"
        )
        
        # Create test_pairs.txt
        with open("AI_Virtual_Wardrobe/Data_preprocessing/test_pairs.txt", "w") as f:
            f.write(f"{img_name} {cloth_name}")
        
        # Run virtual try-on model
        subprocess.run([sys.executable, "AI_Virtual_Wardrobe/test.py"], check=True)
        
        return True
        
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        return False

# === STREAMLIT UI ===
st.title("👗 AI Virtual Try-On")
st.markdown("Upload your image and cloth to see the virtual try-on result.")

# Setup environment
if setup_environment():
    uploaded_img = st.file_uploader("Upload your person image", type=["jpg", "png"], key="person")
    uploaded_cloth = st.file_uploader("Upload your cloth image", type=["jpg", "png"], key="cloth")

    if uploaded_img and uploaded_cloth:
        st.success("Images uploaded. Generating try-on...")
        
        with st.spinner("Processing..."):
            # Save inputs with expected names
            person_path = "AI_Virtual_Wardrobe/inputs/img/000001_0.png"
            cloth_path = "AI_Virtual_Wardrobe/inputs/cloth/000001_1.png"
            
            # Save uploaded files
            with open(person_path, "wb") as f:
                f.write(uploaded_img.read())
            with open(cloth_path, "wb") as f:
                f.write(uploaded_cloth.read())
            
            # Run the pipeline
            if run_pipeline_function():
                # Check for result
                tryon_path = "AI_Virtual_Wardrobe/results/test/try-on/000001_0.png"
                
                # Try different possible result paths
                possible_paths = [
                    "AI_Virtual_Wardrobe/results/test/try-on/000001_0.png",
                    "AI_Virtual_Wardrobe/results/000001_0.png",
                    "results/000001_0.png"
                ]
                
                result_found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        st.image(path, caption="👗 Try-On Result")
                        result_found = True
                        break
                
                if not result_found:
                    st.error("Result not found. Please check the pipeline configuration.")
                    # Show available files for debugging
                    st.write("Available files in results directory:")
                    for root, dirs, files in os.walk("AI_Virtual_Wardrobe"):
                        if "result" in root.lower():
                            st.write(f"Directory: {root}")
                            for file in files:
                                st.write(f"  - {file}")
else:
    st.error("Environment setup failed. Please check the logs and try again.")

