import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model_path = 'models/lungcan_model_efficientnetb0_Finetuned.h5'
model = load_model(model_path)
class_names = ['Benign', 'Malignant', 'Normal']

def preprocess_and_predict(image_path):
    """Load, preprocess, and predict the class of an image."""
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image array

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class

if __name__ == '__main__':
    # Example usage
    # image_path = 'path_to_your_image.jpg'
    # result = preprocess_and_predict(image_path)
    # print(f"Predicted class for the image is: {result}")
