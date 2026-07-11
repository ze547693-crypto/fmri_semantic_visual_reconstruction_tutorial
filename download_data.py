import os
import zipfile
import shutil
import requests
import config


def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading: {save_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {save_path}")


def extract_zip(zip_path, extract_path, password=None):
    if os.path.isdir(os.path.join(extract_path, "training")) and os.path.isdir(os.path.join(extract_path, "test")):
        print("Stimuli already extracted.")
        return

    print("Extracting GOD stimuli...")
    with zipfile.ZipFile(zip_path, "r") as z:
        pwd = password.encode() if password else None
        z.extractall(extract_path, pwd=pwd)


def organize_god_stimuli(base_path):
    images_subfolder = os.path.join(base_path, "images")
    source = images_subfolder if os.path.isdir(images_subfolder) else base_path

    for file_name in ["image_test_id.csv", "image_training_id.csv"]:
        src = os.path.join(source, file_name)
        dst = os.path.join(base_path, file_name)
        if os.path.exists(src) and src != dst:
            shutil.move(src, dst)

    for folder_name in ["training", "test"]:
        src = os.path.join(source, folder_name)
        dst = os.path.join(base_path, folder_name)
        if os.path.exists(src) and src != dst:
            shutil.move(src, dst)

    required = [
        os.path.join(base_path, "training"),
        os.path.join(base_path, "test"),
        os.path.join(base_path, "image_test_id.csv"),
        os.path.join(base_path, "image_training_id.csv"),
    ]

    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file/folder: {p}")

    print("GOD stimuli structure verified.")


def main():
    subject_path = os.path.join(config.GOD_FMRI_PATH, "Subject3.h5")
    stimuli_zip_path = os.path.join(config.GOD_IMAGENET_PATH, "image_stimuli.zip")

    download_file(config.SUBJECT3_URL, subject_path)
    download_file(config.GOD_STIMULI_URL, stimuli_zip_path)
    download_file(config.CLASS_TO_WORDNET_URL, config.CLASS_TO_WORDNET_JSON)

    extract_zip(stimuli_zip_path, config.GOD_IMAGENET_PATH, password="W9kaBybu")
    organize_god_stimuli(config.GOD_IMAGENET_PATH)

    print("All required data is ready.")


if __name__ == "__main__":
    main()