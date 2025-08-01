# clip_classifier.py

import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import io
import fitz  # PyMuPF
import streamlit as st

# THIS IS THE MOST IMPORTANT PART FOR PERFORMANCE
# This Streamlit decorator caches the model in memory. Without it, the app would
# be incredibly slow, as it would reload the 1GB+ model on every interaction.
# This replaces the simple model loading from Cell 3.
@st.cache_resource
def load_model():
    """
    Loads and caches the CLIP model and processor using Streamlit's caching.
    This function will only run once when the app is first started.
    """
    st.write("Cache miss: Loading CLIP model for the first time...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model and processor loaded successfully.")
    return model, processor, device

# This is the classification function from Cell 3, slightly modified
# to accept the model as an argument.
def classify_image(model, processor, device, image: Image, candidate_labels: list):
    """Classifies a single PIL image using the loaded CLIP model."""
    # We must convert the image to RGB format for the model
    rgb_image = image.convert("RGB")
    text_descriptions = [f"A photo of a {label}" for label in candidate_labels]
    inputs = processor(text=text_descriptions, images=rgb_image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    scores = {label: prob for label, prob in zip(candidate_labels, probs)}
    predicted_label = max(scores, key=scores.get)
    return predicted_label

    

    
    
   
def extract_images_from_pdf(uploaded_file):
    """Extracts images from an uploaded PDF file."""
    images = []
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            for img in doc.get_page_images(page_num):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
        doc.close()
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
    return images