import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import config


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = get_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image


class ImageNet256Dataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.idx_to_class = {}

        class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for idx, class_name in enumerate(tqdm(class_names, desc="Scanning ImageNet256")):
            self.idx_to_class[idx] = class_name
            class_dir = os.path.join(root_dir, class_name)

            files = []
            files += glob.glob(os.path.join(class_dir, "*.jpg"))
            files += glob.glob(os.path.join(class_dir, "*.jpeg"))
            files += glob.glob(os.path.join(class_dir, "*.png"))
            files += glob.glob(os.path.join(class_dir, "*.JPEG"))

            for p in files:
                self.image_paths.append(p)
                self.labels.append(idx)

        print("ImageNet256 images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = get_transform()(image)
        label = self.labels[idx]
        return image, label


def load_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(config.DEVICE)
    model.eval()
    return model


@torch.no_grad()
def extract_features_from_paths(image_paths):
    dataset = ImagePathDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = load_resnet50()
    features = []

    for images in tqdm(loader, desc="Extracting ResNet50"):
        images = images.to(config.DEVICE)
        feat = model(images)
        feat = torch.flatten(feat, 1)
        features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    print("Features:", features.shape)
    return features


@torch.no_grad()
def precompute_imagenet256_features():
    feature_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_features_resnet50.npy")
    labels_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_labels.npy")
    class_map_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_idx_to_class.npy")

    if os.path.exists(feature_file) and os.path.exists(labels_file) and os.path.exists(class_map_file):
        print("Loading cached ImageNet256 features...")
        return (
            np.load(feature_file),
            np.load(labels_file),
            np.load(class_map_file, allow_pickle=True).item(),
        )

    dataset = ImageNet256Dataset(config.IMAGENET256_PATH)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = load_resnet50()

    all_features = []
    all_labels = []

    start = time.time()

    for images, labels in tqdm(loader, desc="Extracting ImageNet256 ResNet50"):
        images = images.to(config.DEVICE)
        feat = model(images)
        feat = torch.flatten(feat, 1)

        all_features.append(feat.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    class_map = dataset.idx_to_class

    np.save(feature_file, features)
    np.save(labels_file, labels)
    np.save(class_map_file, class_map)

    print("Saved ImageNet256 features.")
    print("Minutes:", (time.time() - start) / 60)

    return features, labels, class_map