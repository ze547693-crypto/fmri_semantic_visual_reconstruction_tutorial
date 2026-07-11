import numpy as np
from sklearn.neighbors import NearestNeighbors
import config


def build_class_prototypes(db_features, db_labels, db_class_map):
    unique_labels = sorted(np.unique(db_labels).astype(int))

    prototype_features = []
    prototype_readable_labels = []

    for label in unique_labels:
        idx = np.where(db_labels.astype(int) == label)[0]
        class_features = db_features[idx]

        prototype = class_features.mean(axis=0)

        prototype_features.append(prototype)
        prototype_readable_labels.append(db_class_map[int(label)])

    prototype_features = np.vstack(prototype_features)
    prototype_readable_labels = np.array(prototype_readable_labels)

    print("Class prototype features:", prototype_features.shape)

    return prototype_features, prototype_readable_labels


def train_knn(prototype_features):
    knn = NearestNeighbors(
        n_neighbors=config.KNN_MAX_NEIGHBORS,
        metric=config.KNN_METRIC,
        algorithm="brute",
        n_jobs=-1,
    )

    knn.fit(prototype_features)

    return knn


def retrieve_topk(knn, query_embeddings, prototype_readable_labels, prototype_synsets):
    distances, indices = knn.kneighbors(query_embeddings)

    retrieved_readables = prototype_readable_labels[indices]
    retrieved_synsets = prototype_synsets[indices]

    return distances, indices, retrieved_readables, retrieved_synsets


def compute_topk_accuracy(true_synsets, retrieved_synsets, k):
    correct = []

    for true_label, preds in zip(true_synsets, retrieved_synsets):
        correct.append(true_label in preds[:k])

    accuracy = float(np.mean(correct))

    return accuracy, correct