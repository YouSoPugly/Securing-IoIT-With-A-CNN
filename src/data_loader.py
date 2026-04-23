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

FLOOD_SUBTYPES_19 = {
    'ack-frag-flood':     'Ack_Fragmentation_Flood',
    'connect-flood':      'Connect_Flood',
    'http-flood':         'HTTP_Flood',
    'icmp-frag-flood':    'ICMP_Fragmentation_Flood',
    'icmp-flood':         'ICMP_Flood',
    'mqtt-publish-flood': 'MQTT_Publish_Flood',
    'push-ack-flood':     'PSHACK_Flood',
    'rst-fin-flood':      'RSTFIN_Flood',
    'slowloris':          'Slowloris',
    'syn-flood':          'SYN_Flood',
    'synonymousip-flood': 'Synonymous_IP_Flood',
    'tcp-flood':          'TCP_Flood',
    'udp-frag-flood':     'UDP_Fragmentation_Flood',
    'udp-flood':          'UDP_Flood',
}

NON_FLOOD_CATEGORIES_19 = {
    # Recon
    'host-disc-arp-ping':        'Recon-Host_Discovery_ARP_Ping',
    'host-disc-tcp-ack-ping':    'Recon-Host_Discovery_TCP_ACK_Ping',
    'host-disc-tcp-syn-ping':    'Recon-Host_Discovery_TCP_SYN_Ping',
    'host-disc-tcp-syn-stealth': 'Recon-Host_Discovery_TCP_SYN_Stealth',
    'host-disc-udp-ping':        'Recon-Host_Discovery_UDP_Ping',
    'os-scan':                   'Recon-OS_Scan',
    'ping-sweep':                'Recon-Ping_Sweep',
    'port-scan':                 'Recon-Port_Scan',
    'vuln-scan':                 'Recon-VulScan',
    'sql-injection-blind':       'Web-Blind_SQL_Injection',
    'sql-injection':             'Web-SQL_Injection',
    'backdoor-upload':           'Web-Backdoor_Upload',
    'command-injection':         'Web-Command_Injection',
    'xss':                       'Web-XSS',
    # Bruteforce
    'dictionary-ssh':            'Bruteforce-SSH',
    'dictionary-telnet':         'Bruteforce-Telnet',
    # MITM
    'arp-spoofing':              'MITM-ARP_Spoofing',
    'impersonation':             'MITM-Impersonation',
    'ip-spoofing':               'MITM-IP_Spoofing',
    # Malware
    'mirai-syn-flood':           'Malware-Mirai_SYN_Flood',
    'mirai-udp-flood':           'Malware-Mirai_UDP_Flood',
    'malware':                   'Malware-Other',
    # Benign
    'benign':                    'Benign',
}

ATTACK_CATEGORIES_8 = {
    'ddos':       'DDoS',
    'dos':        'DoS',
    'recon':      'Recon',
    'web':        'Web',
    'bruteforce': 'Bruteforce',
    'mitm':       'MITM',
    'malware':    'Malware',
    'attack':     'Malware',
    'benign':     'Benign',
}

ATTACK_CATEGORIES_2 = {
    'attack': 'Attack',
    'benign': 'Benign',
}

def get_attack_category(label, class_config, coarse_label=None):
    key = str(label).strip().lower()

    if class_config in (2, 8):
        categories = ATTACK_CATEGORIES_2 if class_config == 2 else ATTACK_CATEGORIES_8
        if key in categories:
            return categories[key]
        for cat_key, cat_val in categories.items():
            if cat_key in key:
                return cat_val
        return None

    if class_config == 19:
        if key in NON_FLOOD_CATEGORIES_19:
            return NON_FLOOD_CATEGORIES_19[key]
        for cat_key, cat_val in NON_FLOOD_CATEGORIES_19.items():
            if cat_key in key:
                return cat_val

        coarse = str(coarse_label).strip().lower() if coarse_label is not None else ''
        if coarse in ('ddos', 'dos'):
            prefix = 'DDoS' if coarse == 'ddos' else 'DoS'
            for cat_key, cat_val in FLOOD_SUBTYPES_19.items():
                if cat_key in key:
                    return f'{prefix}-{cat_val}'
        return None

    return None

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

    if class_config == 19:
        y_train = train_df.apply(
            lambda row: get_attack_category(row[label_col], 19, row.get('label2')),
            axis=1,
        )
        y_test = test_df.apply(
            lambda row: get_attack_category(row[label_col], 19, row.get('label2')),
            axis=1,
        )
    else:
        y_train = train_df[label_col].apply(lambda x: get_attack_category(x, class_config))
        y_test  = test_df[label_col].apply(lambda x: get_attack_category(x, class_config))


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
