# CS.4371 Group 6 Project
## Overview
This project evaluates the robustness of the 1D-CNN architecture originally proposed for Medical IoT (IoMT) when applied to Industrial IoT (IIoT) environments. 

Following the "archeology of ideas" in this field:
* **Foundational Bedrock:** Our work is built upon the research done by **Mohammadi et al. (2024)** [1], which established the effectiveness of CNNs in detecting threats within healthcare IoT device traffic.
* **Contemporary Context:** We acknowledge and build upon the findings of **Firouzi et al. (2025)** [2], who introduced the **IIoT 2025 Dataset**. This dataset represents the modern landscape of industrial sensor attacks that our implementation aims to mitigate.

We have adapted the original dataloaders to handle the 2025 dataset schema and compared the CNN's performance against traditional regression and ensemble methods (Adaboost, Logistic Regression, and Random Forest).

---

## Project Status & Functionality
* **Dataloader Implementation:** Fully Functional. Supports automated CSV parsing and loadng for the IIoT 2025 dataset.
* **CNN Model:** Fully Functional. Supports Binary (2), Categorical (8), and Multiclass (19) configurations.
* **Comparison Models:** Fully Functional. Implementations for K-Nearest Neighbors, Adaboost, Random Forest, and Logistic Regression are included in the `/src` directory.

---

## Getting Started

### Step 1: Clone Repository
```bash
git clone https://github.com/YouSoPugly/Securing-IoIT-With-A-CNN
```

### Step 2: Install Requirements
Navigate to Project Directory
```bash
cd Securing-IoIT-With-A-CNN
```
then:
```bash
pip install -r requirements.txt
```
> **Note:** We recommend using Python 3.9 for this project

### Step 3: Download Dataset
Download the **IIoT 2025 Dataset** from: [Link-to-Dataset](https://txst-my.sharepoint.com/:u:/g/personal/rii11_txstate_edu/IQDXQaMGbkfnT5b9pquzIxSXAaYTV1ulxJhmxXme1wqAuiE?e=ZHF350)

### Step 4: Prepare Data
Uncompress the .tar.xz file and replace the data/train/ and data/test/ folders with the ones in the uncompressed dataset (iiot2025/train/ iiot2025/test).
```
data/
├── train/     ← Move files from iiot2025/train to data/train
└── test/      ← Move files from iiot2025/test to data/test
```

### Step 5: Run Training
Navigate to the `src` directory:
```bash
cd src
```
To run the CNN model, execute `main.py` and specify the classification configuration:
```bash
python main.py --class_config <num_classes>
```

Replace `<num_classes>` with:
- **2** for binary classification,
- **8** for categorical,
- **19** for multiclass.

**Example (binary classification):**
```bash
python main.py --class_config 2
```
---

## Project Structure

```
project/
├── data/
│   ├── train/                        # Training CSV files
│   └── test/                         # Testing CSV files
├── src/
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── model.py                      # CNN model definition and training
|   ├── train_logistic_regression.py  # Implemetation of Logistic Regression
|   ├── train_knn.py                  # Implemetation of K-Nearest Neighbors
│   └── main.py                       # CNN execution script
├── requirements.txt                  # Project dependencies
└── README.md                         # What you're reading right now :D
```
---

## Citations

[1] Mohammadi, A., Ghahramani, H., Asghari, S. A., & Aminian, M. (2024, October). Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection. In 2024 19th Iranian Conference on Intelligent Systems (ICIS) (pp. 168-173). IEEE.

[2] Firouzi, A.; Dadkhah, S.; Maret, S.A.; Ghorbani, A.A. "DataSense: A Real-Time Sensor-Based Benchmark Dataset for Attack Analysis in IIoT with Multi-Objective Feature Selection." Electronics, 14, 4095, 2025.

---
