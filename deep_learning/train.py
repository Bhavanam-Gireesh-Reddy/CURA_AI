import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import create_model


# Paths

DATASET_PATH = "classification_data"
ARTIFACT_PATH = "artifacts"


# Training parameters

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10


# Create artifacts folder

os.makedirs(
    ARTIFACT_PATH,
    exist_ok=True
)


# Data augmentation

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



# Load training data

train_data = train_datagen.flow_from_directory(

    os.path.join(DATASET_PATH, "train"),

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)



# Load validation data

valid_data = valid_datagen.flow_from_directory(

    os.path.join(DATASET_PATH, "valid"),

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)



# Get class names

class_names = list(
    train_data.class_indices.keys()
)


print(
    "Classes:",
    class_names
)



# Save class names

with open(
    os.path.join(
        ARTIFACT_PATH,
        "class_names.json"
    ),
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )



# Create model

model = create_model(
    len(class_names)
)


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)



model.summary()



# Save best model

checkpoint = ModelCheckpoint(

    filepath=os.path.join(
        ARTIFACT_PATH,
        "best_model.h5"
    ),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)



# Stop if no improvement

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)



# Training

history = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop
    ]

)



# Save training history

with open(
    os.path.join(
        ARTIFACT_PATH,
        "training_history.json"
    ),
    "w"
) as f:

    json.dump(

        history.history,

        f,

        indent=4

    )


print(
    "Training completed successfully!"
)

print(
    "Model saved inside artifacts/"
)