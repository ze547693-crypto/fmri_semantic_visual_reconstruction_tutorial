import os
import torch

KAGGLE_BASE_PATH = "/kaggle/working"

DATA_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, "data")
GOD_FMRI_PATH = os.path.join(DATA_BASE_PATH, "GOD")
GOD_IMAGENET_PATH = os.path.join(DATA_BASE_PATH, "imagenet", "images")

OUTPUT_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, "output")
MODELS_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, "models")

IMAGENET256_PATH = "/kaggle/input/datasets/dimensi0n/imagenet-256"
IMAGENET256_FEATURES_PATH = os.path.join(OUTPUT_BASE_PATH, "imagenet256_features")

CLASS_TO_WORDNET_JSON = os.path.join(KAGGLE_BASE_PATH, "class_to_wordnet.json")

GENERATED_IMAGES_PATH = os.path.join(OUTPUT_BASE_PATH, "generated_images")
RESULTS_PATH = os.path.join(OUTPUT_BASE_PATH, "results")

SUBJECT_ID = "3"
ROI = "ROI_VC"

SUBJECT3_URL = "https://drive.usercontent.google.com/download?id=1lHipXEuAks-3nQ-ZI1n3f9Sicuwa5ko_&export=download&confirm=t&uuid=688e6c59-df8b-42dc-a680-8cb864ff54f0"

GOD_STIMULI_URL = "https://drive.usercontent.google.com/download?id=1Qv80fpgNf43eGeGad6NZnHbgqF3KdP1_&export=download&authuser=0&confirm=t&uuid=df1b776f-147d-4780-808b-b89920cfa355&at=ABswASafR4lGAXaDmZE2XUjdjYxI%3A1782670946192"

CLASS_TO_WORDNET_URL = "https://raw.githubusercontent.com/enomodnara/brain_decoding/refs/heads/main/class_to_wordnet.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2

RIDGE_ALPHAS = [10, 100, 1000, 10000]
CV_FOLDS = 5
RANDOM_STATE = 42

KNN_MAX_NEIGHBORS = 10
KNN_METRIC = "cosine"

STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5
DIFFUSION_STEPS = 25

for path in [
    GOD_FMRI_PATH,
    GOD_IMAGENET_PATH,
    OUTPUT_BASE_PATH,
    MODELS_BASE_PATH,
    IMAGENET256_FEATURES_PATH,
    GENERATED_IMAGES_PATH,
    RESULTS_PATH,
]:
    os.makedirs(path, exist_ok=True)

print("Using device:", DEVICE)