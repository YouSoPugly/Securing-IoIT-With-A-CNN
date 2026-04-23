import argparse
import glob
import os
import re

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = "../data/train/*.csv"
TEST_PATH = "../data/test/*.csv"


# get label from filename
def get_label_name(filepath):
    name = os.path.basename(filepath)

    name = name.replace("_train.pcap.csv", "")
    name = name.replace("_test.pcap.csv", "")
    name = name.replace("_test.pcap", "")
    name = name.replace(".pcap.csv", "")
    name = name.replace(".csv", "")

    name = re.sub(r"\(\d+\)$", "", name)

    return name.strip()


# clean labels so they match the dataset format
def clean_label(label):
    if label == "Benign":
        return "Benign"

    if label == "ARP_Spoofing":
        return "Spoofing"

    # MQTT and Recon already look fine
    if label.startswith("MQTT-") or label.startswith("Recon-"):
        return label

    # collapse TCP_IP variants like TCP_IP-DDoS-ICMP1
    match = re.match(r"TCP_IP-(DDoS|DoS)-(ICMP|SYN|TCP|UDP)\d*$", label)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    return label


# choose label based on class setup
def get_class_label(label, class_config):
    label = clean_label(label)

    if class_config == 19:
        return label

    if class_config == 6:
        if label == "Benign":
            return "Benign"
        if label == "Spoofing":
            return "Spoofing"
        if label.startswith("Recon-"):
            return "Recon"
        if label.startswith("MQTT-"):
            return "MQTT"
        if label.startswith("DDoS-"):
            return "DDoS"
        if label.startswith("DoS-"):
            return "DoS"
        return "Other"

    if class_config == 2:
        if label == "Benign":
            return "Benign"
        return "Attack"

    raise ValueError("class_config must be 2, 6, or 19")


# load all csv files and attach labels
def load_dataset(path_pattern, class_config):
    files = glob.glob(path_pattern)

    if not files:
        raise FileNotFoundError(f"No CSV files found at: {path_pattern}")

    dfs = []

    for file in files:
        df = pd.read_csv(file)

        label_name = get_label_name(file)
        df["Label"] = get_class_label(label_name, class_config)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Random Forest on IoMT dataset")
    parser.add_argument("--class_config", type=int, required=True, choices=[2, 6, 19])
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_test", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading training data ({args.class_config} classes)...")
    train_df = load_dataset(TRAIN_PATH, args.class_config)

    print(f"Loading test data ({args.class_config} classes)...")
    test_df = load_dataset(TEST_PATH, args.class_config)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    if args.sample_train is not None:
        print(f"Sampling {args.sample_train} rows...")
        train_df = train_df.sample(n=min(args.sample_train, len(train_df)), random_state=42)

    if args.sample_test is not None:
        print(f"Sampling {args.sample_test} rows...")
        test_df = test_df.sample(n=min(args.sample_test, len(test_df)), random_state=42)


    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    print("\nFilling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Running predictions...")
    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()