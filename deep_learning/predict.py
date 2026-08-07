import tensorflow as tf
import numpy as np
import json
import sys

from tensorflow.keras.preprocessing import image


MODEL_PATH = "deep_learning/artifacts/fine_tuned_best.keras"
CLASS_PATH = "deep_learning/artifacts/class_names.json"

IMG_SIZE = 224


# Load model

model = tf.keras.models.load_model(
    MODEL_PATH
)


# Load class names

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)



def predict_image(image_path):

    # Load image

    img = image.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )


    # Convert image to array

    img_array = image.img_to_array(
        img
    )


    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Same preprocessing used during training

    img_array = tf.keras.applications.efficientnet.preprocess_input(
        img_array
    )


    # Prediction

    predictions = model.predict(
        img_array
    )


    # Get highest probability class

    predicted_index = np.argmax(
        predictions[0]
    )


    confidence = (
        predictions[0][predicted_index] * 100
    )


    # Final output

    print("\nPrediction Result")
    print("----------------------")

    print(
        "Fracture Type:",
        class_names[predicted_index]
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )
    prediction = model.predict(img_array)

    print("Raw prediction:", prediction)
    print("Sum:", np.sum(prediction[0]))



if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py <image_path>"
        )

        exit()


    image_path = sys.argv[1]


    predict_image(
        image_path
    )