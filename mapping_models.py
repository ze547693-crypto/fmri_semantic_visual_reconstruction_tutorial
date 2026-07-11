import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import config


def train_ridge(X_train, Z_train, force_retrain=False):
    model_path = os.path.join(config.MODELS_BASE_PATH, "mapping_sub3_ridge_resnet50.sav")

    if os.path.exists(model_path) and not force_retrain:
        print("Loading saved Ridge model...")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    print("Training Ridge model...")
    print("X_train:", X_train.shape)
    print("Z_train:", Z_train.shape)

    ridge = Ridge(random_state=config.RANDOM_STATE)

    grid = GridSearchCV(
        ridge,
        {"alpha": config.RIDGE_ALPHAS},
        cv=config.CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )

    grid.fit(X_train, Z_train)
    model = grid.best_estimator_

    print("Best alpha:", grid.best_params_["alpha"])
    print("Best CV R2:", grid.best_score_)

    Z_train_pred = model.predict(X_train)
    print("Train RMSE:", np.sqrt(mean_squared_error(Z_train, Z_train_pred)))
    print("Train R2:", r2_score(Z_train, Z_train_pred))

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Saved:", model_path)

    return model


def predict_embeddings(model, X):
    Z_pred = model.predict(X)
    print("Predicted embeddings:", Z_pred.shape)
    return Z_pred


def adjust_predicted_embeddings(Z_pred, Z_train):
    train_mean = np.mean(Z_train, axis=0)
    train_std = np.std(Z_train, axis=0)

    pred_mean = np.mean(Z_pred, axis=0)
    pred_std = np.std(Z_pred, axis=0)

    return ((Z_pred - pred_mean) / (pred_std + 1e-9)) * train_std + train_mean


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)