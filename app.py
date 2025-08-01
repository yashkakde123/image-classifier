# app.py

import streamlit as st
from PIL import Image

# Import the functions from your backend file
from clip_classifier import (
    load_model,
    classify_image,
    extract_images_from_pdf
)
CONFIDENCE_THRESHOLD = 0.75 # 75% confidence required

# --- Classification Labels ---
# These are the prompts for CLIP.
CANDIDATE_LABELS = [
    "A medical image such as an X-ray, MRI, CT scan, ultrasound, pathology slide, or a photograph from a surgical procedure.",
    "A standard photograph of a person, animal, landscape, city, or an everyday object or scene or any visualization. This is not a scientific or medical image."
]

LABEL_MAP = {
    "A medical image such as an X-ray, MRI, CT scan, ultrasound, pathology slide, or a photograph from a surgical procedure.": "Medical",
    "A standard photograph of a person, animal, landscape, city, or an everyday object or scene or any visualization. This is not a scientific or medical image.": "Non-Medical"
}

# (Moved classification logic to after model is loaded and images are available)

# --- App Configuration ---
st.set_page_config(page_title="GenAI Image Classifier", layout="wide")

# --- App Title and Description ---
st.title("Medical vs. Non-Medical Image Classifier 🩺")
st.markdown("This app uses OpenAI's **CLIP model** for **Zero-Shot Image Classification**.  upload a PDF to extract and classify all images.")

# --- Load Model ---
# The st.cache_resource decorator in the backend ensures this runs only once.
model, processor, device = load_model()

# --- Classification Labels ---
# These are the prompts for CLIP.
CANDIDATE_LABELS = [
    "A medical image such as an X-ray, MRI, CT scan, ultrasound, pathology slide, or a photograph from a surgical procedure.",
    "A standard photograph of a person, animal, landscape, city, or an everyday object. This is not a scientific or medical image."
]

LABEL_MAP = {
    "A medical image such as an X-ray, MRI, CT scan, ultrasound, pathology slide, or a photograph from a surgical procedure.": "Medical",
    "A standard photograph of a person, animal, landscape, city, or an everyday object. This is not a scientific or medical image.": "Non-Medical"
}

# --- User Interface ---
st.sidebar.header("Input")
input_method = st.sidebar.radio("Choose input method:", ("PDF Upload", "Image Upload"))

images_to_classify = []
source_name = ""

if input_method == "PDF Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if st.sidebar.button("Extract Images from PDF"):
        if uploaded_file:
            with st.spinner(f"Extracting images from {uploaded_file.name}..."):
                images_to_classify = extract_images_from_pdf(uploaded_file)
                source_name = uploaded_file.name
        else:
            st.sidebar.warning("Please upload a PDF file.")

elif input_method == "Image Upload":
    uploaded_images = st.sidebar.file_uploader("Upload image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.sidebar.button("Add Images"):
        if uploaded_images:
            for img_file in uploaded_images:
                try:
                    img = Image.open(img_file)
                    images_to_classify.append(img)
                except Exception as e:
                    st.sidebar.warning(f"Could not open image: {img_file.name}. Error: {e}")
            source_name = f"{len(images_to_classify)} uploaded image(s)"
        else:
            st.sidebar.warning("Please upload at least one image.")

# --- Display Results ---
if images_to_classify:
    st.header(f"Found {len(images_to_classify)} images from: {source_name}")
    
    # Display images and classifications in columns for a clean layout
    cols = st.columns(4) 
    for i, img in enumerate(images_to_classify):
        with cols[i % 4]:
            st.image(img, use_container_width=True, caption=f"Image {i+1}")
            try:
                # Classify the image using the backend function
                predicted_desc = classify_image(model, processor, device, img, CANDIDATE_LABELS)
                final_label = LABEL_MAP[predicted_desc]
                
                # Use color to highlight the result
                if final_label == "Medical":
                    st.error(f"**Classification: {final_label}**")
                else:
                    st.info(f"**Classification: {final_label}**")
            except Exception as e:
                st.warning(f"Could not classify image {i+1}. Error: {e}")
else:
    st.info("Waiting for input... Please provide a URL or upload a PDF and click 'Extract'.")