import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

def load_model(model_path='model/model.tflite'):
    # Load the TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_and_predict(image_path, model_interpreter, input_size=(224, 224)):
    # Load and preprocess the image
    img = Image.open(image_path).resize(input_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get input and output tensors
    input_details = model_interpreter.get_input_details()
    output_details = model_interpreter.get_output_details()

    # Set the tensor to point to the input data to be inferred
    model_interpreter.set_tensor(input_details[0]['index'], img_array)
    model_interpreter.invoke()

    # Extract the output
    predictions = model_interpreter.get_tensor(output_details[0]['index'])
    return predictions
