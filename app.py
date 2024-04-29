from flask import Flask, render_template, request, url_for
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load your custom model
model = load_model('model/LungLesNet.h5')

@app.route('/', methods=["GET"])
def load_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    if imagefile:  # Check if an image was actually uploaded
         # Use secure_filename to avoid security issues
        image_filename = os.path.join('images/', secure_filename(imagefile.filename)) 
        full_image_path = os.path.join('static', image_filename)  # Save inside the static directory
        imagefile.save(full_image_path)

        # Image processing for prediction
        image = load_img(full_image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)

        # classes model was trained on
        class_names = ['Benign', 'Malignant', 'Normal']  
        top_pred = np.argmax(yhat)
        confidence = yhat[0][top_pred] * 100  # Convert probability to percentage

        classification = '%s (%.2f%%)' % (class_names[top_pred], confidence)

        # Pass the image path along with the prediction to the template
        return render_template('index.html', prediction=classification, image_path=url_for('static', filename=image_filename))
    else:
        return render_template('index.html', error="No image was uploaded. Please upload an image.")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
