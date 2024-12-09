from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image

print("App is starting...")

# Set up the Flask app
app = Flask(__name__)

# Set Tesseract command path 
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Route to display the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and OCR processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['photo']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read the image from the uploaded file
    img = Image.open(file.stream).convert('RGB')

    # Convert the image to OpenCV format for preprocessing
    cv_image = np.array(img)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR) 

    # Preprocess the iamge (e.g., grayscale, thresholding)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR on the image
    text = pytesseract.image_to_string(thresh)

    return f"<h1>Extracted Text</h1><p>{text}</p>"

# Run the app 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


