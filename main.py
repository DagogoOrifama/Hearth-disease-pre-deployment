import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def preprocess_and_predict(image_path, model_path='model/lungcan_model_efficientnetb0_Finetuned.h5', class_names=['Benign', 'Malignant', 'Normal']):
    # Load the model
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    
    # Determine the predicted class name
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, predictions


#test_prediction =getPrediction('example.jpg')
