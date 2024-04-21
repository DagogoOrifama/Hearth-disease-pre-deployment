import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/lungcan_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the file from post request
    img_file = request.files['file']
    
    if img_file:
        # Preprocess the image
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_name = ['Benign', 'Malignant', 'Normal'][predicted_class[0]]
        
        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_class_name}')

if __name__ == "__main__":
    app.run(debug=True)
