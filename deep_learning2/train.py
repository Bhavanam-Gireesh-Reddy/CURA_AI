import os
import json
import tensorflow as tf

from dataset import load_dataset
from model import build_model


# Create artifacts folder
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Load dataset
train_ds, val_ds, test_ds, class_names = load_dataset()


# Save class names
with open(os.path.join(ARTIFACT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)


# Build model
model = build_model()


# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(ARTIFACT_DIR, "best_mri_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)


# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)

print("\n==========================")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print("==========================")

print("\nModel saved successfully in:")
print("deep_learning2/artifacts/best_mri_model.h5")