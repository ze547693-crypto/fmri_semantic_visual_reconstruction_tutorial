import os
import numpy as np
import pandas as pd
import bdpy
from sklearn.preprocessing import StandardScaler
import config


class GodFmriDataHandler:
    def __init__(self):
        self.h5_file = os.path.join(config.GOD_FMRI_PATH, "Subject3.h5")
        self.test_img_csv = os.path.join(config.GOD_IMAGENET_PATH, "image_test_id.csv")
        self.train_img_csv = os.path.join(config.GOD_IMAGENET_PATH, "image_training_id.csv")

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(self.h5_file)

        self.dat = bdpy.BData(self.h5_file)

        self.test_img_df = pd.read_csv(self.test_img_csv, header=None, names=["id", "filename"])
        self.train_img_df = pd.read_csv(self.train_img_csv, header=None, names=["id", "filename"])

    def load(self):
        voxel_data = self.dat.select(config.ROI)

        metadata = {
            "DataType": self.dat.select("DataType").squeeze(),
            "Run": self.dat.select("Run").astype(int).squeeze(),
            "stimulus_id": self.dat.select("stimulus_id").squeeze(),
        }

        print("Loaded fMRI:", voxel_data.shape)
        return voxel_data, metadata

    def normalize_by_run(self, X, runs):
        X_norm = X.copy()

        for run in np.unique(runs):
            idx = runs == run
            scaler = StandardScaler()
            X_norm[idx] = scaler.fit_transform(X_norm[idx])

        return X_norm

    def map_images(self, metadata):
        train_map = self.train_img_df.set_index("id")["filename"].to_dict()
        test_map = self.test_img_df.set_index("id")["filename"].to_dict()

        image_paths = []

        for dtype, stim_id in zip(metadata["DataType"], metadata["stimulus_id"]):
            if dtype == 1:
                fname = train_map.get(stim_id)
                folder = "training"
            elif dtype in [2, 3]:
                fname = test_map.get(stim_id)
                folder = "test"
            else:
                image_paths.append(None)
                continue

            if fname is None:
                image_paths.append(None)
            else:
                image_paths.append(os.path.join(config.GOD_IMAGENET_PATH, folder, fname))

        return image_paths

    def get_data_splits(self):
        X, metadata = self.load()
        X = self.normalize_by_run(X, metadata["Run"])

        image_paths = self.map_images(metadata)

        train_idx = [
            i for i, t in enumerate(metadata["DataType"])
            if t == 1 and image_paths[i] is not None and os.path.exists(image_paths[i])
        ]

        test_idx = [
            i for i, t in enumerate(metadata["DataType"])
            if t == 2 and image_paths[i] is not None and os.path.exists(image_paths[i])
        ]

        X_train = X[train_idx]
        train_images = [image_paths[i] for i in train_idx]

        X_test_raw = X[test_idx]
        test_images_raw = [image_paths[i] for i in test_idx]
        test_stim_ids = metadata["stimulus_id"][test_idx]

        X_test_avg = []
        test_images_avg = []

        for stim_id in sorted(set(test_stim_ids)):
            local_idx = np.where(test_stim_ids == stim_id)[0]
            X_test_avg.append(X_test_raw[local_idx].mean(axis=0))
            test_images_avg.append(test_images_raw[local_idx[0]])

        X_test_avg = np.array(X_test_avg)

        print("Train:", X_train.shape)
        print("Test averaged:", X_test_avg.shape)

        return X_train, train_images, X_test_avg, test_images_avg