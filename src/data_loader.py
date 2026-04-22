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

ATTACK_CATEGORIES_19 = {
    # DDoS
    'ack-fragmentation-flood': 'DDoS-Ack_Fragmentation_Flood',
    'connect-flood':           'DDoS-Connect_Flood',
    'http-flood':              'DDoS-HTTP_Flood',
    'icmp-flood':              'DDoS-ICMP_Flood',
    'icmp-fragmentation-flood':'DDoS-ICMP_Fragmentation_Flood',
    'mqtt-publish-flood':      'DDoS-MQTT_Publish_Flood',
    'pshack-flood':            'DDoS-PSHACK_Flood',
    'rstfin-flood':            'DDoS-RSTFIN_Flood',
    'slowloris':               'DDoS-Slowloris',
    'syn-flood':               'DDoS-SYN_Flood',
    'synonymous-ip-flood':     'DDoS-Synonymous_IP_Flood',
    'tcp-flood':               'DDoS-TCP_Flood',
    'udp-flood':               'DDoS-UDP_Flood',
    'udp-fragmentation-flood': 'DDoS-UDP_Fragmentation_Flood',
    # DoS (shares same sub-attack names, distinguished by label2)
    'dos-ack-fragmentation-flood': 'DoS-Ack_Fragmentation_Flood',
    'dos-connect-flood':           'DoS-Connect_Flood',
    'dos-http-flood':              'DoS-HTTP_Flood',
    'dos-icmp-flood':              'DoS-ICMP_Flood',
    'dos-icmp-fragmentation-flood':'DoS-ICMP_Fragmentation_Flood',
    'dos-mqtt-publish-flood':      'DoS-MQTT_Publish_Flood',
    'dos-pshack-flood':            'DoS-PSHACK_Flood',
    'dos-rstfin-flood':            'DoS-RSTFIN_Flood',
    'dos-slowloris':               'DoS-Slowloris',
    'dos-syn-flood':               'DoS-SYN_Flood',
    'dos-synonymous-ip-flood':     'DoS-Synonymous_IP_Flood',
    'dos-tcp-flood':               'DoS-TCP_Flood',
    'dos-udp-flood':               'DoS-UDP_Flood',
    'dos-udp-fragmentation-flood': 'DoS-UDP_Fragmentation_Flood',
    # Recon
    'host-discovery-arp-ping':     'Recon-Host_Discovery_ARP_Ping',
    'host-discovery-tcp-ack-ping': 'Recon-Host_Discovery_TCP_ACK_Ping',
    'host-discovery-tcp-syn-ping': 'Recon-Host_Discovery_TCP_SYN_Ping',
    'host-discovery-tcp-syn-stealth': 'Recon-Host_Discovery_TCP_SYN_Stealth',
    'host-discovery-udp-ping':     'Recon-Host_Discovery_UDP_Ping',
    'os-scan':                     'Recon-OS_Scan',
    'ping-sweep':                  'Recon-Ping_Sweep',
    'port-scan':                   'Recon-Port_Scan',
    'vulnerability-scan':          'Recon-VulScan',
    # Web
    'backdoor-upload':             'Web-Backdoor_Upload',
    'command-injection':           'Web-Command_Injection',
    'sql-injection':               'Web-SQL_Injection',
    'blind-sql-injection':         'Web-Blind_SQL_Injection',
    'cross-site-scripting':        'Web-XSS',
    # Bruteforce
    'ssh-bruteforce':              'Bruteforce-SSH',
    'telnet-bruteforce':           'Bruteforce-Telnet',
    # MITM
    'arp-spoofing':                'MITM-ARP_Spoofing',
    'impersonation':               'MITM-Impersonation',
    'ip-spoofing':                 'MITM-IP_Spoofing',
    # Malware
    'mirai-syn-flood':             'Malware-Mirai_SYN_Flood',
    'mirai-udp-flood':             'Malware-Mirai_UDP_Flood',
    # Benign
    'benign':                      'Benign',
}

ATTACK_CATEGORIES_8 = {
    'ddos':       'DDoS',
    'dos':        'DoS',
    'recon':      'Recon',
    'web':        'Web',
    'bruteforce': 'Bruteforce',
    'mitm':       'MITM',
    'malware':    'Malware',
    'benign':     'Benign',
}

ATTACK_CATEGORIES_2 = {
    'attack': 'Attack',
    'benign': 'Benign',
}

def get_attack_category(label, class_config):
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 8:
        categories = ATTACK_CATEGORIES_8
    else:  # 19
        categories = ATTACK_CATEGORIES_19

    key = str(label).strip().lower()

    if key in categories:
        return categories[key]

    for cat_key in categories:
        if cat_key in key:
            return categories[cat_key]

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
