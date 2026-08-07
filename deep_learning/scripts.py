import tensorflow as tf
import numpy as np
import json
import os

from tensorflow.keras.preprocessing import image


# Base directory of deep_learning

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "fine_tuned_model.keras"
)


CLASS_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "class_names.json"
)


IMG_SIZE = 224


# Load model once

model = tf.keras.models.load_model(
    MODEL_PATH
)


# Load class names

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)



def predict_fracture(image_path):

    """
    Predict fracture from X-ray image

    Returns:
    {
        "fracture_type": "Wrist Fracture",
        "confidence": 95.4
    }
    """

    # Load image

    img = image.load_img(
        image_path,
        target_size=(
            IMG_SIZE,
            IMG_SIZE
        )
    )


    # Convert image

    img_array = image.img_to_array(
        img
    )


    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # EfficientNet preprocessing

    img_array = tf.keras.applications.efficientnet.preprocess_input(
        img_array
    )


    # Prediction

    prediction = model.predict(
        img_array
    )


    predicted_index = np.argmax(
        prediction[0]
    )


    confidence = float(
        prediction[0][predicted_index] * 100
    )


    return {
    
        "fracture_type": class_names[predicted_index].replace("_", " ").title(),
        "confidence": round(confidence,2)
    
}