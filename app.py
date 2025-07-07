import streamlit as st
import os
import sys
import subprocess
from PIL import Image

# === SETUP ===
def setup_environment():
    try:
        st.write("📁 Creating required folders...")

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
            "AI_Virtual_Wardrobe/inputs/cloth"
        ]
        for d in subdirs:
            os.makedirs(d, exist_ok=True)

        return True
    except Exception as e:
        st.error(f"Setup failed: {str(e)}")
        return False

# === PIPELINE ===
def run_pipeline_function():
    try:
        sys.path.append("u2net")
        sys.path.append("AI_Virtual_Wardrobe")

        import u2net_load, u2net_run
        from predict_pose import generate_pose_keypoints

        img_name = "000001_0.png"
        cloth_name = "000001_1.png"
        img_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
        cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"

        st.write("🖼 Loading and resizing uploaded images...")
        img = Image.open(img_path).resize((192, 256), Image.BICUBIC)
        cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")
        img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
        cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")

        st.write("🧠 Loading and running U2Net...")
        u2net = u2net_load.model("u2netp")
        u2net_run.infer(u2net,
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
                        "AI_Virtual_Wardrobe/Data_preprocessing/test_edge")

        st.write("🧍 Running human parsing...")
        result = subprocess.run([
            sys.executable, "Parsing-/simple_extractor.py",
            "--dataset", "lip",
            "--model-restore", "AI_Virtual_Wardrobe/lip_final.pth",
            "--input-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
            "--output-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_label"
        ], capture_output=True, text=True)
        st.text(result.stdout)
        st.text(result.stderr)
        if result.returncode != 0:
            st.error("❌ Human parsing failed.")
            return False

        st.write("🕺 Generating pose keypoints...")
        generate_pose_keypoints(
            img_path,
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_pose/{img_name.replace('.png', '_keypoints.json')}"
        )

        st.write("📋 Writing test_pairs.txt...")
        with open("AI_Virtual_Wardrobe/Data_preprocessing/test_pairs.txt", "w") as f:
            f.write(f"{img_name} {cloth_name}")

        st.write("🚀 Running final try-on test...")
        result = subprocess.run([
            sys.executable, "AI_Virtual_Wardrobe/test.py"
        ], capture_output=True, text=True)

        st.text(result.stdout)
        st.text(result.stderr)

        if result.returncode != 0:
            st.error("❌ Final try-on failed.")
            return False

        st.success("✅ Try-on completed.")
        return True

    except Exception as e:
        st.error(f"Pipeline crashed: {str(e)}")
        return False

# === STREAMLIT UI ===
st.title("👗 AI Virtual Try-On")
st.markdown("Upload your image and clothing to generate a virtual try-on result.")

# Manual setup once
if st.button("🔧 Setup Environment"):
    with st.spinner("Setting up..."):
        if setup_environment():
            st.success("✅ Setup complete.")
        else:
            st.error("❌ Setup failed.")

uploaded_img = st.file_uploader("📷 Upload person image", type=["jpg", "png"])
uploaded_cloth = st.file_uploader("👕 Upload cloth image", type=["jpg", "png"])

if uploaded_img and uploaded_cloth:
    if st.button("🚀 Generate Try-On"):
        with st.spinner("Running the virtual try-on pipeline..."):
            with open("AI_Virtual_Wardrobe/inputs/img/000001_0.png", "wb") as f:
                f.write(uploaded_img.read())
            with open("AI_Virtual_Wardrobe/inputs/cloth/000001_1.png", "wb") as f:
                f.write(uploaded_cloth.read())

            if run_pipeline_function():
                result_path = "results/test/try-on/000001_0.png"
                if os.path.exists(result_path):
                    st.image(result_path, caption="👗 Try-On Result")
                else:
                    st.error("❌ Result not found.")



