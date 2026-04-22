import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

META_COLUMNS = [
    'device_name', 'device_mac',
    'label_full', 'label1', 'label2', 'label3', 'label4',
    'timestamp', 'timestamp_start', 'timestamp_end'
]

LABEL_COLUMN = {
    2:  'label1',
    8:  'label2',
    19: 'label3'
}

def load_and_preprocess_data(data_dir, class_config):
    if class_config not in LABEL_COLUMN:
        raise ValueError(f"class_config must be 2, 8, or 19. Got: {class_config}")

    label_col = LABEL_COLUMN[class_config]

    def _load_split(split: str) -> pd.DataFrame:
        split_dir = os.path.join(data_dir, split)
        files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.csv')
        ]
        if not files:
            raise FileNotFoundError(f"No CSV files found in: {split_dir}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    train_df = _load_split('train')
    test_df  = _load_split('test')

    y_train = train_df[label_col].astype(str).str.strip().str.lower()
    y_test  = test_df[label_col].astype(str).str.strip().str.lower()

    X_train = train_df.drop(columns=[c for c in META_COLUMNS if c in train_df.columns])
    X_test  = test_df.drop(columns=[c for c in META_COLUMNS if c in test_df.columns])

    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
    X_train = X_train.drop(columns=non_numeric_cols)
    X_test  = X_test.drop(columns=[c for c in non_numeric_cols if c in X_test.columns])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded  = label_encoder.transform(y_test)

    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical  = to_categorical(y_test_encoded)

    X_train, X_val, y_train_categorical, y_val_categorical = train_test_split(
        X_train, y_train_categorical, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val   = X_val.reshape(X_val.shape[0],   X_val.shape[1],   1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder
