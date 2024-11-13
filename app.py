from flask import Flask, request, render_template, send_file
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded")

        image = request.files['image']
        accuracy = request.form.get('accuracy')

        if image.filename == '':
            return render_template('index.html', error="No file selected")

        if image:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # Compress the image
            compressed_filename = f"compressed_{image.filename}"
            compressed_path = os.path.join(app.config['COMPRESSED_FOLDER'], compressed_filename)
            reduce_image(image_path, float(accuracy), compressed_path)

            # Provide the download option
            return render_template('index.html', download_url=f"/download/{compressed_filename}")

    return render_template('index.html', download_url=None)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['COMPRESSED_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

def reduce_image(file_name, accuracy, output_path):
    # Step 1: Load the original image
    image = io.imread(file_name)
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
    print(f"Image successfully compressed and saved at {output_path}")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['COMPRESSED_FOLDER']):
        os.makedirs(app.config['COMPRESSED_FOLDER'])
    
    app.run(host='0.0.0.0', port=5050)