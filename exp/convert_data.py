"""Converting data to CVAT Format."""

import csv
import os

from otx.algorithms.action.utils.convert_public_data_to_cvat import convert_action_cls_dataset_to_datumaro

ANNOT_FILE = "./data/annotation.csv"


def time_conversion(input_string):
    """Time conversion function."""
    if input_string == "N/A":
        return input_string
    h, m, s = input_string.split(":")
    return 3600 * int(h) + 60 * int(m) + int(s)


def convert_data():
    """Convert data to CVAT format.

    Expected data structure
    - data
        - frames
            - vid1
                - image_00001.jpg
                - image_00002.jpg
                - image_00003.jpg
            - vid2
                - image_00001.jpg
                - image_00002.jpg
                - image_00003.jpg
        - annotation.csv

    Output data structure
    - data
        - cvat
            - train
                - vid1
                    - images
                        - 00001.jpg
                        - 00002.jpg
                        - 00003.jpg
                    - annotations.xml
                - vid2
                    - images
                        - 00001.jpg
                        - 00002.jpg
                        - 00003.jpg
                    - annotations.xml
    """
    print("Converting to intermedate data format")
    os.makedirs("./data/intermediate/")
    with (
            open(ANNOT_FILE) as input_file,
            open("./data/intermediate/train.txt", "w") as train_txt,
            open("./data/intermediate/val.txt", "w") as val_txt
    ):
        reader = csv.DictReader(input_file)
        os.makedirs("./data/intermediate/rawframes/")
        for row in reader:
            video_file_name = row["Weld video File name"]
            video_start_time = time_conversion(row["Weld Start Time"])
            abnormal_start_time = time_conversion(row["Weld fault Start time"])
            video_end_time = time_conversion(row["Weld End time"])
            row["Weld fault"]
            split = row["Split"]
            if not os.path.exists(f"./data/frames/{video_file_name}"):
                continue
            if abnormal_start_time == "N/A":
                src = f"./data/frames/{video_file_name}"
                dst = f"./data/intermediate/rawframes/{video_file_name}"
                os.system(f"cp -r {src} {dst}")
                if split == "training":
                    train_txt.write(f"{video_file_name} {video_end_time * 30} 0\n")
                else:
                    val_txt.write(f"{video_file_name} {video_end_time * 30} 0\n")
            else:
                start_frame = video_start_time * 30
                end_frame = video_end_time * 30
                abnormal_start_frame = abnormal_start_time * 30
                os.makedirs(f"./data/intermediate/rawframes/{video_file_name}_0")
                for seq in range(1, start_frame):
                    src = f"./data/frames/{video_file_name}/image_{seq:05d}.jpg"
                    dst = f"./data/intermediate/rawframes/{video_file_name}_0/image_{seq:05d}.jpg"
                    os.system(f"cp {src} {dst}")
                if split == "training":
                    train_txt.write(f"{video_file_name}_0 {len(range(1, start_frame))}, 0\n")
                else:
                    val_txt.write(f"{video_file_name}_0 {len(range(1, start_frame))}, 0\n")

                os.makedirs(f"./data/intermediate/rawframes/{video_file_name}_1")
                for seq in range(start_frame, abnormal_start_frame):
                    src = f"./data/frames/{video_file_name}/image_{seq:05d}.jpg"
                    dst = f"./data/intermediate/rawframes/{video_file_name}_1/image_{seq:05d}.jpg"
                    os.system(f"cp {src} {dst}")
                if split == "training":
                    train_txt.write(f"{video_file_name}_1 {len(range(start_frame, abnormal_start_frame))}, 1\n")
                else:
                    val_txt.write(f"{video_file_name}_1 {len(range(start_frame, abnormal_start_frame))}, 1\n")

                os.makedirs(f"./data/intermediate/rawframes/{video_file_name}_2")
                for seq in range(abnormal_start_frame, end_frame + 1):
                    src = f"./data/frames/{video_file_name}/image_{seq:05d}.jpg"
                    dst = f"./data/intermediate/rawframes/{video_file_name}_2/image_{seq:05d}.jpg"
                    os.system(f"cp {src} {dst}")
                if split == "training":
                    train_txt.write(f"{video_file_name}_2 {len(range(abnormal_start_frame, end_frame))}, 2\n")
                else:
                    val_txt.write(f"{video_file_name}_2 {len(range(abnormal_start_frame, end_frame))}, 2\n")

    with open("./data/intermediate/label_map.txt", "w") as label_txt:
        label_txt.write("no action\n")
        label_txt.write("abnormal\n")
        label_txt.write("normal\n")

    print("Done")
    print("Converting to CVAT format")

    convert_action_cls_dataset_to_datumaro(
        "./data/intermediate/rawframes",
        "./data/cvat/train",
        "./data/intermediate/train.txt",
        "./data/intermediate/label_map.txt"
    )
    convert_action_cls_dataset_to_datumaro(
        "./data/intermediate/rawframes",
        "./data/cvat/val",
        "./data/intermediate/val.txt",
        "./data/intermediate/label_map.txt"
    )
    os.system("rm -rf ./data/intermediate/")
    print("Done")


if __name__ == "__main__":
    convert_data()
