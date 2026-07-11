# 🌿 Crop Disease Detection — EfficientNetB0 Transfer Learning

A deep learning pipeline that classifies **crop disease categories** using **EfficientNetB0** with transfer learning. The model leverages features pre-trained on ImageNet and fine-tunes them on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) to detect diseases from leaf images.

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
  - [Run on Google Colab (Recommended)](#run-on-google-colab-recommended)
  - [Run Locally](#run-locally)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Results & Inference](#-results--inference)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

- **Transfer Learning** — Uses EfficientNetB0 (pre-trained on ImageNet) for fast, accurate feature extraction.
- **Data Augmentation** — Applies random flips, rotations, and zoom to improve model robustness.
- **Fine-Tuning** — Unfreezes the base model after initial training for domain-specific optimization.
- **End-to-End Pipeline** — Includes model training, evaluation, and inference with visualization.
- **Kaggle Integration** — Automatically downloads the PlantVillage dataset from Kaggle.
- **Google Drive Integration** — Saves trained model and results to Google Drive for persistence.
- **Colab Optimized** — Designed to run on Google Colab with GPU support.

---

## 📁 Project Structure

```
crop disease detection/
├── crop_disease_detection.py    # Main training & inference script (optimized for Colab)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ✅ Prerequisites

| Requirement       | Details                                               |
| :----------------- | :---------------------------------------------------- |
| **Python**         | 3.8+ (3.10+ recommended)                              |
| **TensorFlow**     | ≥ 2.13.0                                              |
| **GPU (optional)** | NVIDIA GPU with CUDA 12+ (for faster training)        |
| **RAM**            | 8 GB minimum (16 GB recommended)                      |
| **Disk Space**     | ~3 GB for the PlantVillage dataset + model weights    |
| **Kaggle Account** | Required to download the dataset automatically         |

---

## � Quick Start

### Run on Google Colab (Recommended)

This script is **optimized for Google Colab** where Kaggle API is readily available. Follow these steps:

#### Step 1: Set Up Kaggle Authentication
1. Go to [kaggle.com](https://www.kaggle.com/settings/account)
2. Click **Create New Token** — this downloads `kaggle.json`
3. In Colab, the script will prompt you to upload this file

#### Step 2: Run the Script
1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or upload this script as a `.py` file
3. Copy and paste the contents of `crop_disease_detection.py` into a Colab cell, or upload the file
4. **Set Runtime to GPU**: Click `Runtime → Change runtime type → T4 GPU`
5. Run the cells in order

#### Step 3: Script Workflow
The script automatically:
- ✅ Installs Kaggle
- ✅ Downloads the PlantVillage dataset (~3 GB)
- ✅ Trains the EfficientNetB0 model on the dataset
- ✅ Performs fine-tuning with unfrozen base layers
- ✅ Generates accuracy and loss plots
- ✅ Creates confusion matrix and classification report
- ✅ Saves the trained model to Google Drive
- ✅ Loads the model and performs inference on a sample image

#### Step 4: Access Your Results
After training completes, all results are saved to:
```
Google Drive → My Drive → crop_disease_project/
├── model.keras                  # Trained model
├── class_names.json            # Class labels
├── history.json                # Training history
├── accuracy_plot.png           # Accuracy curve
├── loss_plot.png               # Loss curve
├── classification_report.txt   # Model metrics
└── confusion_matrix.npy        # Confusion matrix
```

---

### Run Locally

To run locally on your machine, you need to **modify the script**:

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Set Up Kaggle
```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
# Place it in ~/.kaggle/
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Modify the Script
Remove or comment out these Colab-specific lines:
```python
# Remove these lines:
!pip install -q kaggle
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip plantdisease.zip -d data
```

Instead, download the dataset manually:
```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data
```

#### Step 4: Modify Google Drive Paths
Replace this:
```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_PATH = '/content/drive/MyDrive/crop_disease_project/'
```

With a local directory:
```python
import os
SAVE_PATH = './results/'
os.makedirs(SAVE_PATH, exist_ok=True)
```

#### Step 5: Run the Script
```bash
python crop_disease_detection.py
```

---

## 📂 Dataset

The script uses the **PlantVillage dataset** with ~54,000 images of healthy and diseased crop leaves across 38 classes:
- Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tomato
- Each crop has multiple disease variants (e.g., "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy")

---

## 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
        │
Data Augmentation (flip, rotate, zoom)
        │
  EfficientNetB0 (ImageNet weights, initially frozen)
        │
  GlobalAveragePooling2D
        │
  BatchNormalization
        │
    Dense (128, ReLU)
        │
    Dropout (0.5)
        │
  Dense (num_classes, Softmax)
        │
   Output: Class probabilities
```

| Component                 | Details                                                       |
| :------------------------ | :------------------------------------------------------------ |
| **Base Model**            | EfficientNetB0 (pre-trained on ImageNet, ~237 layers)         |
| **Data Augmentation**     | RandomFlip, RandomRotation, RandomZoom                        |
| **Pooling**               | GlobalAveragePooling2D — reduces feature maps to vectors      |
| **Hidden Layer**          | Dense (128 units, ReLU) with Batch Normalization              |
| **Regularization**        | Dropout at 50% to prevent overfitting                         |
| **Classifier**            | Dense layer with Softmax activation for multi-class output    |

---

## 📈 Training Pipeline

The training is split into two phases:

### Phase 1 — Initial Training (Feature Extraction)
- The EfficientNetB0 base is **frozen** (weights are not updated).
- Only the custom classifier head is trained.
- **Optimizer:** Adam (learning rate = `1e-4`)
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 10
- **Batch Size:** 32

**Result:** The model learns to recognize disease patterns using pre-trained ImageNet features.

### Phase 2 — Fine-Tuning
- The EfficientNetB0 base model is **unfrozen** (all layers can be updated).
- The entire model is re-trained with a much smaller learning rate.
- **Optimizer:** Adam (learning rate = `1e-5`)
- **Loss:** Sparse Categorical Crossentropy
- **Additional Epochs:** 5
- **Batch Size:** 32

**Result:** The model adapts pre-trained features to better recognize crop diseases.

### Data Split
- **Training:** 80% of the dataset
- **Validation:** 20% of the dataset
- **Seed:** 123 (for reproducibility)

---

## 🗂 Results & Inference

### Saved Artifacts
After training completes, the script saves:

| File                          | Description                                              |
| :---------------------------- | :------------------------------------------------------- |
| `model.keras`                 | Trained and fine-tuned model weights                     |
| `class_names.json`            | Disease class labels (38 categories)                     |
| `history.json`                | Training accuracy, loss, validation metrics              |
| `accuracy_plot.png`           | Training vs. validation accuracy curve                   |
| `loss_plot.png`               | Training vs. validation loss curve                       |
| `classification_report.txt`   | Precision, recall, F1-score per class                    |
| `confusion_matrix.npy`        | Confusion matrix for error analysis                      |

### Inference on New Images
The script includes a complete inference pipeline:

```python
# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Preprocess image
img = image.load_img('path/to/leaf_image.jpg', target_size=(224, 224))
img_array = preprocess_input(image.img_to_array(img))
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Predicted Disease: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# Top 3 predictions
top3 = np.argsort(predictions[0])[-3:][::-1]
for i in top3:
    print(f"{class_names[i]}: {predictions[0][i]:.3f}")
```

### Evaluation Metrics
The script generates:
- **Confusion Matrix:** Shows which classes are confused with each other
- **Classification Report:** Precision, recall, and F1-scores per disease class
- **Training Curves:** Visualization of accuracy and loss during training

---

## 🔧 Troubleshooting

| Issue | Solution |
| :---- | :------- |
| **`ModuleNotFoundError: No module named 'tensorflow'`** | Run `pip install tensorflow>=2.13.0` or use Google Colab. |
| **`kaggle: command not found`** | Run `pip install kaggle` (already in requirements.txt). |
| **Kaggle API authentication fails** | Ensure `kaggle.json` is uploaded and placed in `~/.kaggle/` with permissions 600. |
| **Out of Memory (OOM) errors** | Reduce `BATCH_SIZE` in the script (from 32 to 16 or 8). |
| **Slow training on CPU** | Use Google Colab with GPU runtime (`Runtime → Change runtime type → T4 GPU`). |
| **`FileNotFoundError: data/plantvillage not found`** | Ensure the dataset is downloaded: `kaggle datasets download -d emmarex/plantdisease && unzip plantdisease.zip -d data`. |
| **Low accuracy** | Try training for more epochs, increasing data augmentation, or using a larger batch size. |
| **`Could not load dynamic library 'cudart64_...'`** | CUDA is not installed. Use CPU-only mode or install CUDA Toolkit 12+ for GPU training. |
| **Google Drive mounting fails** | Ensure you are running in Google Colab and have authenticated your Google account. |
| **Model not found during inference** | Check that the model file path is correct and the model was saved successfully. |

---

## 📜 License

This project is for educational and research purposes. The PlantVillage dataset is publicly available for research use.
