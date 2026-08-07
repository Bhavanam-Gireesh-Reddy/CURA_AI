import os
import shutil
from pathlib import Path


DATASET_ROOT = "xray_data"
OUTPUT_ROOT = "classification_data"

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png"
]


def get_number_of_classes():

    classes = set()

    for split in ["train", "valid", "test"]:

        label_path = os.path.join(
            DATASET_ROOT,
            split,
            "labels"
        )

        for file in os.listdir(label_path):

            if file.endswith(".txt"):

                with open(
                    os.path.join(label_path, file),
                    "r"
                ) as f:

                    lines = f.readlines()

                    for line in lines:

                        class_id = int(line.split()[0])
                        classes.add(class_id)

    return sorted(classes)



def create_folders(classes):

    for split in ["train", "valid", "test"]:

        for cls in classes:

            os.makedirs(
                os.path.join(
                    OUTPUT_ROOT,
                    split,
                    f"class{cls}"
                ),
                exist_ok=True
            )



def find_image(folder, name):

    for ext in IMAGE_EXTENSIONS:

        path = os.path.join(
            folder,
            name + ext
        )

        if os.path.exists(path):
            return path

    return None



def process_split(split):

    print(f"Processing {split}")

    images_folder = os.path.join(
        DATASET_ROOT,
        split,
        "images"
    )

    labels_folder = os.path.join(
        DATASET_ROOT,
        split,
        "labels"
    )


    for label_file in os.listdir(labels_folder):

        if not label_file.endswith(".txt"):
            continue


        label_path = os.path.join(
            labels_folder,
            label_file
        )


        with open(label_path,"r") as f:
            lines=f.readlines()


        if len(lines)==0:
            continue


        # Take first object class
        class_id=int(
            lines[0].split()[0]
        )


        image_name=Path(label_file).stem


        image_path=find_image(
            images_folder,
            image_name
        )


        if image_path is None:
            print(
                "Image missing:",
                image_name
            )
            continue


        destination=os.path.join(
            OUTPUT_ROOT,
            split,
            f"class{class_id}",
            os.path.basename(image_path)
        )


        shutil.copy(
            image_path,
            destination
        )


def main():

    classes=get_number_of_classes()

    print(
        "Detected Classes:",
        classes
    )


    create_folders(classes)


    for split in [
        "train",
        "valid",
        "test"
    ]:

        process_split(split)


    print(
        "\nDataset converted successfully"
    )


if __name__=="__main__":
    main()