import os
import numpy as np
import pandas as pd

import config
from data_loading import GodFmriDataHandler
from feature_extraction import extract_features_from_paths, precompute_imagenet256_features
from mapping_models import (
    train_ridge,
    predict_embeddings,
    adjust_predicted_embeddings,
    l2_normalize,
)
from retrieval import (
    build_class_prototypes,
    train_knn,
    retrieve_topk,
    compute_topk_accuracy,
)
from wordnet_utils import (
    build_wordnet_maps,
    extract_synset_from_path,
    readable_label_to_synset,
    make_prompt,
)
from generation import generate_images_from_prompts


def main():
    print("=" * 80)
    print("Subj3 + ROI_VC + ResNet50 + Ridge + Retrieval + Generation")
    print("=" * 80)

    foldername_to_synset, synset_to_readable = build_wordnet_maps(
        config.CLASS_TO_WORDNET_JSON
    )

    handler = GodFmriDataHandler()
    X_train, train_image_paths, X_test, test_image_paths = handler.get_data_splits()

    print("\nExtracting ResNet50 features for GOD train images...")
    Z_train = extract_features_from_paths(train_image_paths)

    print("\nExtracting ResNet50 features for GOD test images...")
    Z_test_true = extract_features_from_paths(test_image_paths)

    mapping_model = train_ridge(X_train, Z_train)

    Z_pred = predict_embeddings(mapping_model, X_test)

    Z_pred_adjusted = adjust_predicted_embeddings(Z_pred, Z_train)
    query_embeddings = l2_normalize(Z_pred_adjusted)

    db_features, db_labels, db_class_map = precompute_imagenet256_features()
    db_features = l2_normalize(db_features)

    prototype_features, prototype_readable_labels = build_class_prototypes(
        db_features,
        db_labels,
        db_class_map,
    )

    prototype_features = l2_normalize(prototype_features)

    prototype_synsets = np.array([
        readable_label_to_synset(label, foldername_to_synset)
        for label in prototype_readable_labels
    ])

    knn = train_knn(prototype_features)

    distances, indices, retrieved_readables, retrieved_synsets = retrieve_topk(
        knn,
        query_embeddings,
        prototype_readable_labels,
        prototype_synsets,
    )

    true_synsets = [extract_synset_from_path(p) for p in test_image_paths]
    true_readables = [synset_to_readable.get(s, "unknown") for s in true_synsets]

    top1_acc, top1_correct = compute_topk_accuracy(true_synsets, retrieved_synsets, 1)
    top5_acc, top5_correct = compute_topk_accuracy(true_synsets, retrieved_synsets, 5)

    print("\nRetrieval Results:")
    print("Top-1 Accuracy:", top1_acc)
    print("Top-5 Accuracy:", top5_acc)

    top1_labels = [labels[0] for labels in retrieved_readables]
    prompts = [make_prompt(label) for label in top1_labels]

    generated_dir = os.path.join(
        config.GENERATED_IMAGES_PATH,
        "sub3_resnet50_generation"
    )

    generated_paths = generate_images_from_prompts(
        prompts,
        generated_dir
    )

    rows = []

    for i in range(len(test_image_paths)):
        rows.append({
            "sample_index": i,
            "ground_truth_path": test_image_paths[i],
            "true_synset": true_synsets[i],
            "true_readable_label": true_readables[i],

            "top1_pred_synset": retrieved_synsets[i][0],
            "top1_pred_readable_label": retrieved_readables[i][0],
            "top1_distance": distances[i][0],
            "top1_correct": bool(top1_correct[i]),

            "top5_pred_synsets": "|".join(retrieved_synsets[i][:5]),
            "top5_pred_readable_labels": "|".join(retrieved_readables[i][:5]),
            "top5_correct": bool(top5_correct[i]),

            "prompt": prompts[i],
            "generated_image_path": generated_paths[i],
        })

    results_df = pd.DataFrame(rows)

    results_csv = os.path.join(
        config.RESULTS_PATH,
        "sub3_resnet50_retrieval_generation_results.csv"
    )

    results_df.to_csv(results_csv, index=False)

    print("\nSaved results:", results_csv)
    print("Generated images:", generated_dir)
    print(results_df.head())


if __name__ == "__main__":
    main()