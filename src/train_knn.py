import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_and_preprocess_data


def flatten_features(X):
    """
    Convert data from shape (n_samples, n_features, 1) to (n_samples, n_features).
    If X is already 2D, return it unchanged.
    """
    if len(X.shape) == 3:
        return X.reshape(X.shape[0], X.shape[1])
    return X


def one_hot_to_labels(y):
    """
    Convert one-hot encoded labels to class indices.
    """
    return np.argmax(y, axis=1)


# Best Score: 0.862 {'metric': 'minkowski', 'n_neighbors': 4, 'weights': 'distance'}
# def train_knn_with_resume(
#     X_train,
#     y_train,
#     checkpoint_path="knn_checkpoint.json",
#     cv=5,
# ):
#     """
#     Manual grid search for KNN with:
#     - progress bar
#     - live best score display
#     - pause/resume via checkpoint file
#     """
#     param_grid = {
#         "n_neighbors": [4],
#         "weights": ["distance"],
#         "metric": ["minkowski"],
#     }
#
#     grid = list(ParameterGrid(param_grid))
#     total = len(grid)
#
#     start_idx = 0
#     best_score = -1.0
#     best_params = None
#
#     # Resume from checkpoint if it exists
#     if os.path.exists(checkpoint_path):
#         try:
#             with open(checkpoint_path, "r") as f:
#                 checkpoint = json.load(f)
#
#             start_idx = int(checkpoint.get("last_index", -1)) + 1
#             best_score = float(checkpoint.get("best_score", -1.0))
#             best_params = checkpoint.get("best_params", None)
#
#             print(f"Resuming from checkpoint: {checkpoint_path}")
#             print(f"Starting at combination {start_idx + 1} of {total}")
#             if best_params is not None:
#                 print(f"Current best score: {best_score:.6f}")
#                 print(f"Current best params: {best_params}")
#         except Exception as e:
#             print(f"Could not load checkpoint ({e}). Starting fresh.")
#             start_idx = 0
#             best_score = -1.0
#             best_params = None
#
#     try:
#         with tqdm(total=total, initial=start_idx, desc="Grid Search", unit="combo") as pbar:
#             for i in range(start_idx, total):
#                 params = grid[i]
#                 model = KNeighborsClassifier(**params)
#
#                 scores = cross_val_score(
#                     model,
#                     X_train,
#                     y_train,
#                     cv=cv,
#                     scoring="f1_weighted",
#                     n_jobs=-1,
#                 )
#
#                 mean_score = float(np.mean(scores))
#
#                 improved = mean_score > best_score
#                 if improved:
#                     best_score = mean_score
#                     best_params = params
#                     print(
#                         f"\nNew best at combo {i + 1}/{total}: "
#                         f"score={best_score:.6f}, params={best_params}"
#                     )
#
#                 # Update live progress bar with the current and best scores
#                 pbar.set_postfix(
#                     current_score=f"{mean_score:.6f}",
#                     best_score=f"{best_score:.6f}",
#                 )
#                 pbar.update(1)
#
#                 # Save checkpoint after each parameter set
#                 checkpoint = {
#                     "last_index": i,
#                     "best_score": best_score,
#                     "best_params": best_params,
#                 }
#                 with open(checkpoint_path, "w") as f:
#                     json.dump(checkpoint, f)
#
#     except KeyboardInterrupt:
#         print("\nPaused. Progress was saved to checkpoint.")
#         print(f"Run again to resume from: {checkpoint_path}")
#         return None, None
#
#     if best_params is None:
#         raise RuntimeError("Grid search finished but no best parameters were found.")
#
#     print("\nGrid search complete.")
#     print("Best Params:", best_params)
#     print("Best CV Score:", best_score)
#
#     final_model = KNeighborsClassifier(**best_params)
#     final_model.fit(X_train, y_train)
#
#     # Remove checkpoint after successful completion
#     if os.path.exists(checkpoint_path):
#         os.remove(checkpoint_path)
#
#     return final_model, best_params


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on test data.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nTest Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred),
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a KNN classifier for network intrusion detection."
    )
    parser.add_argument(
        "--class_config",
        type=int,
        choices=[2, 8, 19],
        default=19,
        help="Number of classes for classification (2, 8, or 19)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory. If omitted, uses ../data relative to this file.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "..", "data")

    X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder = load_and_preprocess_data(
        data_dir, args.class_config
    )

    # KNN expects 2D input
    X_train = flatten_features(X_train)
    X_val = flatten_features(X_val)
    X_test = flatten_features(X_test)

    # Convert one-hot labels to integer labels
    y_train = one_hot_to_labels(y_train_categorical)
    y_val = one_hot_to_labels(y_val_categorical)
    y_test = one_hot_to_labels(y_test_categorical)

    # Use train + val for model selection
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    print(f"Training samples: {X_full_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    model = KNeighborsClassifier(n_neighbors=4, weights="distance", metric="minkowski")
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, label_encoder)