import argparse
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_and_preprocess_data


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression on IIoT dataset")
    parser.add_argument("--class_config", type=int, required=True, choices=[2, 8, 19])
    args = parser.parse_args()

    print(f"Loading and preprocessing data ({args.class_config} classes)...")
    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, label_encoder = load_and_preprocess_data(
        DATA_DIR, args.class_config
    )

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # CNN loader returns 3D arrays: (samples, features, 1)
    # Logistic regression needs 2D: (samples, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # one-hot labels to class ids
    y_train = np.argmax(y_train_cat, axis=1)
    y_val = np.argmax(y_val_cat, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)

    # combine train and val for logistic regression
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_full, y_train_full)

    print("Running predictions...")
    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
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