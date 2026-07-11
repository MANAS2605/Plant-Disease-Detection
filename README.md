# 🌿 Crop Disease Detection — MobileNetV2 Transfer Learning

A deep learning pipeline that classifies **38 crop disease categories** using **MobileNetV2** with transfer learning. The model leverages features pre-trained on ImageNet and fine-tunes them on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) to detect diseases from leaf images.

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Dataset Preparation](#-dataset-preparation)
- [Usage](#-usage)
  - [Run Locally](#option-1-run-locally)
  - [Run on Google Colab](#option-2-run-on-google-colab)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Results & Saved Models](#-results--saved-models)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

- **Transfer Learning** — Uses MobileNetV2 (pre-trained on ImageNet) for fast, accurate feature extraction.
- **Fine-Tuning** — Optionally unfreezes the top 20 layers of MobileNetV2 for domain-specific tuning.
- **Lightweight** — MobileNetV2 is optimized for mobile/edge deployment.
- **Colab Ready** — Includes a Jupyter Notebook designed to run directly on Google Colab with GPU support.
- **Model Checkpointing** — Automatically saves the best model based on validation accuracy.

---

## 📁 Project Structure

```
crop disease detection/
├── mobilenetv2_transfer_learning.py   # Main training script (local)
├── crop_disease_model.ipynb           # Google Colab notebook
├── code_explanation.md                # Beginner-friendly code walkthrough
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## ✅ Prerequisites

| Requirement       | Details                                               |
| :----------------- | :---------------------------------------------------- |
| **Python**         | 3.9 – 3.12 recommended                               |
| **TensorFlow**     | ≥ 2.15.0                                              |
| **GPU (optional)** | NVIDIA GPU with CUDA 12+ for faster training          |
| **RAM**            | 8 GB minimum (16 GB recommended)                      |
| **Disk Space**     | ~3 GB for the PlantVillage dataset + model weights    |

---

## 🛠 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "crop disease detection"
```

### 2. Create a Virtual Environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs TensorFlow ≥ 2.15.0. If you have an NVIDIA GPU, TensorFlow 2.15+ includes GPU support by default — no separate `tensorflow-gpu` package is needed.

---

## 📂 Dataset Preparation

This project is designed for the **PlantVillage** dataset (38 classes of healthy and diseased crop leaves).

### Download the Dataset

1. Download from [Kaggle – PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Extract the archive and organize it in the following structure:

```
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── Apple___Black_rot/
│   ├── ...
│   └── Tomato___healthy/
└── val/
    ├── Apple___Apple_scab/
    ├── ...
    └── Tomato___healthy/
```

> **Tip:** Use `tf.keras.utils.image_dataset_from_directory()` to load images directly from this folder structure. Example:
>
> ```python
> train_dataset = tf.keras.utils.image_dataset_from_directory(
>     'dataset/train',
>     image_size=(224, 224),
>     batch_size=32,
>     label_mode='categorical'
> )
> ```

---

## 🚀 Usage

### Option 1: Run Locally

```bash
python mobilenetv2_transfer_learning.py
```

**What it does:**
1. Builds the MobileNetV2 feature extraction model (frozen base + custom classifier head).
2. Prints the model summary.
3. Prepares the fine-tuning architecture (unfreezes top 20 layers).
4. Prints the updated model summary.

> ⚠️ **Important:** The actual `model.fit()` training calls are commented out in the script because they require you to supply your own dataset (`train_dataset` and `validation_dataset`). Uncomment the training blocks (lines 128–134 and 142–158) and plug in your loaded datasets to start training.

### Option 2: Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `crop_disease_model.ipynb` notebook (or use **File → Open Notebook → Upload**).
3. Set the runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`.
4. Run all cells with **Shift + Enter**.

The notebook verifies that TensorFlow is working, builds the model, and prints the architecture summary — all within Colab's cloud environment.

---

## 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
        │
  MobileNetV2 (ImageNet weights, frozen)
        │
  GlobalAveragePooling2D
        │
    Dropout (0.2)
        │
  Dense (38 units, Softmax)
        │
   Output: 38 class probabilities
```

| Component                 | Details                                                       |
| :------------------------ | :------------------------------------------------------------ |
| **Base Model**            | MobileNetV2 (pre-trained on ImageNet, ~154 layers)            |
| **Pooling**               | GlobalAveragePooling2D — reduces feature maps to 1D vectors   |
| **Regularization**        | Dropout at 20% rate to prevent overfitting                    |
| **Classifier**            | Dense layer with Softmax activation for 38-class output       |

---

## 📈 Training Pipeline

The training is split into two phases:

### Phase 1 — Feature Extraction
- The MobileNetV2 base is **frozen** (no weight updates).
- Only the custom classifier head is trained.
- **Optimizer:** Adam (learning rate = `1e-4`)
- **Loss:** Categorical Crossentropy
- **Epochs:** 10 (recommended)

### Phase 2 — Fine-Tuning
- The **top 20 layers** of MobileNetV2 are unfrozen.
- The entire model is re-trained with a much smaller learning rate.
- **Optimizer:** Adam (learning rate = `1e-5`)
- **Loss:** Categorical Crossentropy
- **Additional Epochs:** 5 (recommended, using `initial_epoch` to continue from Phase 1)

### Model Checkpointing

The best-performing model is saved automatically based on `val_accuracy`:

| Phase             | Saved File                                        |
| :---------------- | :------------------------------------------------ |
| Feature Extraction | `best_mobilenetv2_crop_disease.keras`             |
| Fine-Tuning       | `best_mobilenetv2_crop_disease_finetuned.keras`   |

---

## 🗂 Results & Saved Models

After training, the saved `.keras` model files can be loaded for inference:

```python
from tensorflow.keras.models import load_model

model = load_model('best_mobilenetv2_crop_disease_finetuned.keras')

# Predict on a single image
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img = load_img('path/to/leaf_image.jpg', target_size=(224, 224))
img_array = np.expand_dims(img_to_array(img), axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Predicted class index: {predicted_class}, Confidence: {confidence:.2%}")
```

---

## 🔧 Troubleshooting

| Issue | Solution |
| :---- | :------- |
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow>=2.15.0` or use the Colab notebook. |
| **Out of Memory (OOM) errors** | Reduce `batch_size` (e.g., from 32 to 16 or 8). |
| **Slow training on CPU** | Use Google Colab with a GPU runtime or install CUDA locally. |
| **Low accuracy** | Ensure your dataset is properly split and images are 224×224. Try more epochs or data augmentation. |
| **`Could not load dynamic library 'cudart64_...'`** | CUDA is not installed. Install CUDA Toolkit 12+ or use CPU-only mode. |

---

## 📜 License

This project is for educational and research purposes.

---

> For a beginner-friendly explanation of the code, see [`code_explanation.md`](code_explanation.md).
