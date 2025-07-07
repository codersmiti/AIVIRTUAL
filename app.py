import streamlit as st
import os
import sys
import subprocess
from shutil import move
from PIL import Image
import psutil


# === SETUP ===
def setup_environment():
    try:
        st.write("🔁 Checking and cloning repositories...")

        if not os.path.exists("AI_Virtual_Wardrobe"):
            os.system("git clone https://github.com/jayneel-shah18/AI_Virtual_Wardrobe.git")
        if not os.path.exists("Parsing-"):
            os.system("git clone https://github.com/codersmiti/Parsing-.git")
        if not os.path.exists("u2net"):
            os.system("git clone https://github.com/jayneel-shah18/u2net.git")

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
            "AI_Virtual_Wardrobe/inputs/cloth",
            "u2net/saved_models/u2netp",
            "AI_Virtual_Wardrobe/pose",
            "AI_Virtual_Wardrobe/checkpoints/label2city"
        ]
        for d in subdirs:
            os.makedirs(d, exist_ok=True)

        st.write("📦 Moving model files into place...")
        model_files = {
            "u2netp.pth": "u2net/saved_models/u2netp/u2netp.pth",
            "pose_iter_440000.caffemodel": "AI_Virtual_Wardrobe/pose/pose_iter_440000.caffemodel",
            "exp-schp-201908261155-lip.pth": "AI_Virtual_Wardrobe/lip_final.pth",
            "ACGPN_checkpoints/label2city/latest_net_G.pth": "AI_Virtual_Wardrobe/checkpoints/label2city/latest_net_G.pth",
            "ACGPN_checkpoints/label2city/latest_net_G1.pth": "AI_Virtual_Wardrobe/checkpoints/label2city/latest_net_G1.pth",
            "ACGPN_checkpoints/label2city/latest_net_G2.pth": "AI_Virtual_Wardrobe/checkpoints/label2city/latest_net_G2.pth",
            "ACGPN_checkpoints/label2city/latest_net_U.pth": "AI_Virtual_Wardrobe/checkpoints/label2city/latest_net_U.pth",
            "ACGPN_checkpoints/label2city/opt.txt": "AI_Virtual_Wardrobe/checkpoints/label2city/opt.txt",
        }
        for src, dest in model_files.items():
            if os.path.exists(src) and not os.path.exists(dest):
                move(src, dest)

        return True
    except Exception as e:
        st.error(f"Setup failed: {str(e)}")
        return False


def run_pipeline_function():
    try:
        st.write("🧱 Step 1: Appending sys paths...")
        sys.path.append("u2net")
        sys.path.append("AI_Virtual_Wardrobe")

        st.write("📦 Step 2: Importing model modules...")
        from predict_pose import generate_pose_keypoints
        import u2net_load, u2net_run

        img_name = "000001_0.png"
        cloth_name = "000001_1.png"

        st.write("🖼 Step 3: Loading uploaded images...")
        img_path = f"AI_Virtual_Wardrobe/inputs/img/{img_name}"
        cloth_path = f"AI_Virtual_Wardrobe/inputs/cloth/{cloth_name}"

        img = Image.open(img_path).resize((192, 256), Image.BICUBIC)
        cloth = Image.open(cloth_path).resize((192, 256), Image.BICUBIC).convert("RGB")

        img.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}")
        cloth.save(f"AI_Virtual_Wardrobe/Data_preprocessing/test_color/{cloth_name}")
        st.write("✅ Step 4: Images saved and resized.")

        st.write("🧠 Step 5: Loading U2Net model...")
        u2net = u2net_load.model("u2netp")

        st.write("🎯 Step 6: Running U2Net inference...")
        u2net_run.infer(
            u2net,
            "AI_Virtual_Wardrobe/Data_preprocessing/test_color",
            "AI_Virtual_Wardrobe/Data_preprocessing/test_edge"
        )
        st.write("✅ U2Net inference complete.")

        st.write("🧍 Step 7: Running human parsing script...")
        result = subprocess.run([
            sys.executable, "Parsing-/simple_extractor.py",
            "--dataset", "lip",
            "--model-restore", "AI_Virtual_Wardrobe/lip_final.pth",
            "--input-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_img",
            "--output-dir", "AI_Virtual_Wardrobe/Data_preprocessing/test_label"
        ], capture_output=True, text=True)

        st.write("🔍 Parsing STDOUT:", result.stdout)
        st.write("❌ Parsing STDERR:", result.stderr)
        if result.returncode != 0:
            st.error("❌ Human parsing failed.")
            return False

        st.write("✅ Human parsing complete.")

        st.write("🕺 Step 8: Generating pose keypoints...")
        generate_pose_keypoints(
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_img/{img_name}",
            f"AI_Virtual_Wardrobe/Data_preprocessing/test_pose/{img_name.replace('.png', '_keypoints.json')}"
        )
        st.write("✅ Pose keypoints saved.")

        st.write("📋 Step 9: Writing test_pairs.txt...")
        with open("AI_Virtual_Wardrobe/Data_preprocessing/test_pairs.txt", "w") as f:
            f.write(f"{img_name} {cloth_name}")

        st.write("🧪 Step 10: Running final try-on test...")
        test_py_path = os.path.join("AI_Virtual_Wardrobe", "test.py")
        st.write("🛠 Command:", f"{sys.executable} {test_py_path}")

        mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        st.write(f"🔍 App memory usage before test.py: {mem:.2f} MB")



        result = subprocess.run(
            [sys.executable, test_py_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )

        if result.stdout:
            st.text("📤 Try-On STDOUT:\n" + result.stdout)
        if result.stderr:
            st.text("⚠️ Try-On STDERR:\n" + result.stderr)

        if result.returncode != 0:
            st.error("❌ Final try-on test failed.")
            return False

        st.write("✅ Try-on pipeline completed successfully.")
        return True

    except subprocess.TimeoutExpired:
        st.error("🚨 Try-on step timed out. Model might be too heavy for this environment.")
        return False
    except Exception as e:
        st.error(f"🚨 Pipeline crashed: {str(e)}")
        return False


# === STREAMLIT UI ===
st.title("👗 AI Virtual Try-On")
st.markdown("Upload your image and clothing to generate a virtual try-on result.")

# Auto-setup if not present
if st.button("🔧 Setup Environment (Required First Time)") or not os.path.exists("AI_Virtual_Wardrobe"):
    with st.spinner("Setting up..."):
        success = setup_environment()
        if success:
            st.success("✅ Setup complete.")
        else:
            st.error("❌ Setup failed. See logs above.")

uploaded_img = st.file_uploader("📷 Upload your person image", type=["jpg", "png"])
uploaded_cloth = st.file_uploader("👕 Upload your cloth image", type=["jpg", "png"])

if uploaded_img and uploaded_cloth:
    if st.button("🚀 Generate Try-On"):
        with st.spinner("Running the virtual try-on model..."):
            person_path = "AI_Virtual_Wardrobe/inputs/img/000001_0.png"
            cloth_path = "AI_Virtual_Wardrobe/inputs/cloth/000001_1.png"

            with open(person_path, "wb") as f:
                f.write(uploaded_img.read())
            with open(cloth_path, "wb") as f:
                f.write(uploaded_cloth.read())

            if run_pipeline_function():
                tryon_path = "results/test/try-on/000001_0.png"
                if os.path.exists(tryon_path):
                    st.image(tryon_path, caption="👗 Try-On Result")
                else:
                    st.error("❌ Try-on result not found.")
            else:
                st.error("❌ Try-on pipeline failed. See logs above.")


