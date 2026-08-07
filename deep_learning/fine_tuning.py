import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


MODEL_PATH = "deep_learning/artifacts/best_model.h5"

DATASET_PATH = "deep_learning/classification_data"

ARTIFACT_PATH = "deep_learning/artifacts"

IMG_SIZE = 224
BATCH_SIZE = 32


# -----------------------------
# Load saved model
# -----------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# -----------------------------
# Fine tune EfficientNet
# -----------------------------

base_model = model.layers[0]

# Unfreeze backbone

base_model.trainable = True


# Freeze first layers

for layer in base_model.layers[:-50]:
    layer.trainable = False


# Keep BatchNorm stable

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False



# -----------------------------
# Data preprocessing
# Same as train.py
# -----------------------------

train_datagen = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True
)


valid_datagen = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input

)



train_data = train_datagen.flow_from_directory(

    os.path.join(DATASET_PATH,"train"),

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True
)



valid_data = valid_datagen.flow_from_directory(

    os.path.join(DATASET_PATH,"valid"),

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)



print("Classes:")
print(train_data.class_indices)



# -----------------------------
# Compile
# -----------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=[
        "accuracy"
    ]
)



# -----------------------------
# Callbacks
# -----------------------------

callbacks = [

    ModelCheckpoint(

        filepath=os.path.join(
            ARTIFACT_PATH,
            "fine_tuned_best.keras"
        ),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),


    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1
    ),


    EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True
    )

]



# -----------------------------
# Fine tuning
# -----------------------------

history = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=15,

    callbacks=callbacks

)



print("Fine tuning completed")


# Final save

model.save(
    os.path.join(
        ARTIFACT_PATH,
        "fine_tuned_model.keras"
    )
)


print("Model saved")