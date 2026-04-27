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
    if len(X.shape) == 3:
        return X.reshape(X.shape[0], X.shape[1])
    return X


def one_hot_to_labels(y):
    return np.argmax(y, axis=1)


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
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

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