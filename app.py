import streamlit as st
import os
import sys
import subprocess
import zipfile
import shutil
from PIL import Image
import gdown

@st.cache_resource
def setup_environment():
    try:
        if not os.path.exists("AI_Virtual_Wardrobe"):
            os.system("git clone https://github.com/jayneel-shah18/AI_Virtual_Wardrobe.git")
        if not os.path.exists("Parsing-"):
            os.system("git clone https://github.com/codersmiti/Parsing-.git")
        if not os.path.exists("u2net"):
            os.system("git clone https://github.com/jayneel-shah18/u2net.git")

        u2net_model_path = "u2net/saved_models/u2netp/u2netp.pth"
        if not os.path.exists(u2net_model_path):
            os.makedirs("u2net/saved_models/u2netp", exist_ok=True)
            gdown.download("https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy", u2net_model_path, quiet=True)

        checkpoint_zip = "AI_Virtual_Wardrobe/checkpoints/ACGPN_checkpoints.zip"
        extract_dir = "AI_Virtual_Wardrobe/checkpoints"
        if not os.path.exists(os.path.join(extract_dir, "latest_net_G.pth")):
            os.makedirs(extract_dir, exist_ok=True)
            gdown.download("https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx", checkpoint_zip, quiet=True)
            with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(checkpoint_zip)

        parsing_model_path = "AI_Virtual_Wardrobe/lip_final.pth"
        if not os.path.exists(parsing_model_path):
            gdown.download("https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH", parsing_model_path, quiet=True)

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

        return True
    except Exception as e:
        st.error(f"Setup failed: {str(e)}")
        return False


def run_pipeline_function():
    try:
        sys.path.append("u2net")
        sys.path.append("AI_Virtual_Wardrobe")

        import u2net_load, u2net_run
        from predict_pose import generate_pose_keypoints

        pose_model_path = "AI_Virtual_Wardrobe/pose/pose_iter_440000.caffemodel"
        if not os.path.exists(pose_model_path):
            os.makedirs("AI_Virtual_Wardrobe/pose", exist_ok=True)
            gdown.download(id="1bcsYvDtZMdF-P8rLSmr8qQ-NvjR-4Fd5", output=pose_model_path, quiet=True)

        img_name = "000001_0.png"
        cloth_name = "000001_1.png"
        img_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
        cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"

        img = Image.open(img_path).resize((192, 256), Image.BICUBIC)
        cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")
        img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
        cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")

        u2net = u2net_load.model("u2netp")
        u2net_run.infer(u2net,
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_edge")

        result = subprocess.run([
            sys.executable, "Parsing-/simple_extractor.py",
            "--dataset", "lip",
            "--model-restore", "AI_Virtual_Wardrobe/lip_final.pth",
            "--input-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
            "--output-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_label"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return False

        generate_pose_keypoints(
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}",
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_pose/{img_name.replace('.png', '_keypoints.json')}"
        )

        with open("AI_Virtual_Wardrobe/Data_preprocessing/test_pairs.txt", "w") as f:
            f.write(f"{img_name} {cloth_name}")

        result = subprocess.run([sys.executable, "AI_Virtual_Wardrobe/test.py"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            return False

        # ✅ Space cleanup
        shutil.rmtree("AI_Virtual_Wardrobe/Data_preprocessing", ignore_errors=True)
        shutil.rmtree("results/test/try-on/cloth", ignore_errors=True)
        shutil.rmtree("__pycache__", ignore_errors=True)

        return True

    except Exception as e:
        return False


# === STREAMLIT UI ===
st.title("👗 AI Virtual Try-On")
st.markdown("Upload your person and cloth image to generate a virtual try-on!")

if setup_environment():
    uploaded_img = st.file_uploader("Upload your person image", type=["jpg", "png"])
    uploaded_cloth = st.file_uploader("Upload your cloth image", type=["jpg", "png"])

    if uploaded_img and uploaded_cloth:
        st.success("Files uploaded. Generating result...")

        with st.spinner("Processing..."):
            person_path = "AI_Virtual_Wardrobe/inputs/img/000001_0.png"
            cloth_path = "AI_Virtual_Wardrobe/inputs/cloth/000001_1.png"

            with open(person_path, "wb") as f:
                f.write(uploaded_img.read())
            with open(cloth_path, "wb") as f:
                f.write(uploaded_cloth.read())

            if run_pipeline_function():
                tryon_path = "results/test/try-on/test_label/000001_0.png"
                if os.path.exists(tryon_path):
                    st.image(tryon_path, caption="👗 Try-On Result", use_column_width=True)
                else:
                    st.error("Try-on image not found.")
            else:
                st.error("⚠️ Pipeline failed during execution.")
else:
    st.error("❌ Environment setup failed.")

