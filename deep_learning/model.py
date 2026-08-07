import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(num_classes):

    base_model = tf.keras.applications.EfficientNetB0(

        weights="imagenet",

        include_top=False,

        input_shape=(224,224,3)

    )


    # Enable fine tuning

    base_model.trainable = False


    # Freeze first layers
    for layer in base_model.layers[:-50]:

        layer.trainable = False



    model = models.Sequential([


        base_model,


        layers.GlobalAveragePooling2D(),


        layers.BatchNormalization(),


        layers.Dropout(0.4),


        layers.Dense(

            256,

            activation="relu"

        ),


        layers.Dropout(0.3),


        layers.Dense(

            num_classes,

            activation="softmax"

        )

    ])


    return model