import streamlit as st
import cv2
import numpy as np
import os
import zipfile
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# -------- LOAD DATA --------
data = []
image_paths = []
image_refs = []

base_dir = os.path.dirname(os.path.abspath(__file__))
preferred_folder = os.path.join(base_dir, "dataset", "training_set")
fallback_folder = os.path.join(base_dir, "dataset")
home_dir = os.path.expanduser("~")
downloads_folder = os.path.join(home_dir, "Downloads")

downloads_training_folder = os.path.join(
    downloads_folder, "archive.zip", "dog vs cat", "dataset", "training_set"
)
downloads_zip_path = os.path.join(downloads_folder, "archive.zip")

folder_candidates = [preferred_folder, downloads_training_folder, fallback_folder]

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def add_training_image(img_bgr, label, ref):
    if img_bgr is None:
        return

    img_gray = cv2.resize(img_bgr, (64, 64))
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    data.append(img_gray.flatten())
    image_paths.append(label)
    image_refs.append(ref)


for folder in folder_candidates:
    if not os.path.isdir(folder):
        continue

    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file.lower())[1] not in valid_extensions:
                continue

            path = os.path.join(root, file)

            try:
                img = cv2.imread(path)
                add_training_image(img, path, path)
            except Exception:
                pass


if os.path.isfile(downloads_zip_path):
    try:
        with zipfile.ZipFile(downloads_zip_path, "r") as zf:
            prefix = "dog vs cat/dataset/training_set/"

            for member in zf.namelist():
                lower_member = member.lower()
                ext = os.path.splitext(lower_member)[1]

                if not lower_member.startswith(prefix) or ext not in valid_extensions:
                    continue

                file_bytes = zf.read(member)
                arr = np.frombuffer(file_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                ref = f"zip|{downloads_zip_path}|{member}"
                label = f"{downloads_zip_path}:{member}"
                add_training_image(img, label, ref)
    except Exception:
        pass

data = np.array(data)

# -------- EARLY VALIDATION --------
if len(data) == 0:
    st.error(
        "No training images found. Supported sources: 'dataset/training_set', "
        "'Downloads/archive.zip/dog vs cat/dataset/training_set', or image files in 'dataset'."
    )
    st.stop()

# -------- PCA --------
n_components = min(50, data.shape[0], data.shape[1])
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(data)

# -------- SEARCH FUNCTION --------
def search_similar(img):
    if img is None:
        return []

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vec = img.flatten()
    query_pca = pca.transform([vec])

    similarities = cosine_similarity(query_pca, reduced_data)[0]
    indices = np.argsort(similarities)[::-1]

    return indices[:5]


def load_result_image(ref):
    if ref.startswith("zip|"):
        _, zip_path, member = ref.split("|", 2)
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_bytes = zf.read(member)
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return cv2.imread(ref)

# -------- UI --------
st.title("Image Search Engine")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    st.write("Processing...")

    indices = search_similar(img)

    st.write("Results:")

    for i in indices:
        result_img = load_result_image(image_refs[i])
        if result_img is None:
            continue

        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption=image_paths[i])