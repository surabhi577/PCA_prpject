import streamlit as st
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Set up directories for image storage
UPLOAD_FOLDER = 'uploads/'
COMPRESSED_FOLDER = 'compressed/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(COMPRESSED_FOLDER):
    os.makedirs(COMPRESSED_FOLDER)

# Function to reduce image using PCA
def reduce_image(file_path, accuracy, output_path):
    # Step 1: Load the original image
    image = io.imread(file_path)
    gray_image = color.rgb2gray(image)

    # Step 3: Apply PCA for Dimensionality Reduction
    pca = PCA(n_components=accuracy)
    transformed_image = pca.fit_transform(gray_image)

    # Step 4: Reconstruct the Compressed Image
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and Save the Compressed Image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    io.imsave(output_path, compressed_image_uint8)
    return output_path

# Streamlit UI
st.title("Image Compression with PCA")
st.write("Upload an image, select a compression accuracy, and download the compressed image.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Compression accuracy input
accuracy = st.selectbox("Select PCA components for compression accuracy", options=[0.8, 0.9, 0.95, 0.99])

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Compress the image
    compressed_filename = f"compressed_{uploaded_file.name}"
    compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
    reduce_image(file_path, accuracy, compressed_path)

    # Display compressed image
    st.image(compressed_path, caption="Compressed Image")

    # Download compressed image
    with open(compressed_path, "rb") as f:
        compressed_data = f.read()
    st.download_button("Download Compressed Image", compressed_data, file_name=compressed_filename)
