import streamlit as st
import os
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

# Function to compress image with PCA
def reduce_image(file, accuracy, output_path):
    # Step 1: Load the original image
    image = io.imread(file)
    gray_image = color.rgb2gray(image)
    
    # Step 2: Apply PCA for Dimensionality Reduction
    pca = PCA(n_components=accuracy)
    transformed_image = pca.fit_transform(gray_image)
    
    # Step 3: Reconstruct the Compressed Image
    reconstructed_image = pca.inverse_transform(transformed_image)
    
    # Normalize and Save the Compressed Image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    io.imsave(output_path, compressed_image_uint8)

# Streamlit app interface
st.markdown(
    """
    <style>
        body {
            background-color: #1a1a1d;
            color: #e6e6e6
