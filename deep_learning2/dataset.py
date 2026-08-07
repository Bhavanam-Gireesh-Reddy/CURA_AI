import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

TRAIN_DIR = "deep_learning2/data/Training"
TEST_DIR = "deep_learning2/data/Testing"


def load_dataset():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical"
    )

    class_names = train_dataset.class_names

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, class_names


if __name__ == "__main__":

    train_ds, val_ds, test_ds, classes = load_dataset()

    print("\nClasses:", classes)
    print("Training batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Testing batches:", len(test_ds))