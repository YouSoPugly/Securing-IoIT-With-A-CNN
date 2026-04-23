import argparse
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_and_preprocess_data


DATA_DIR = "../data"


def maybe_sample(X, y, sample_size, random_state=42):
    if sample_size is None or sample_size >= len(X):
        return X, y

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=sample_size, replace=False)
    return X[indices], y[indices]


def main():
    parser = argparse.ArgumentParser(description="Random Forest on IoMT dataset")
    parser.add_argument("--class_config", type=int, required=True, choices=[2, 8, 19])
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_test", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading and preprocessing data ({args.class_config} classes)...")
    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, label_encoder = (
        load_and_preprocess_data(DATA_DIR, args.class_config)
    )

    # RandomForest expects 2D features, but data_loader reshapes to 3D for CNNs
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Convert one-hot labels back to integer class ids
    y_train = np.argmax(y_train_cat, axis=1)
    y_val = np.argmax(y_val_cat, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    if args.sample_train is not None:
        print(f"Sampling {args.sample_train} training rows...")
        X_train, y_train = maybe_sample(X_train, y_train, args.sample_train)

    if args.sample_test is not None:
        print(f"Sampling {args.sample_test} test rows...")
        X_test, y_test = maybe_sample(X_test, y_test, args.sample_test)

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Running predictions...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()