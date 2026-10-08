#  Surface Crack Detection using CNN

## 📌 Project Description

This project uses a **Convolutional Neural Network (CNN)** to detect whether an industrial surface image contains a **Crack** or **No Crack**.

The project performs image preprocessing, dataset splitting, data augmentation, CNN model training, evaluation, and prediction on individual images.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* Pillow
* CNN (Convolutional Neural Network)

## 📂 Project Structure

```text
MarvellousCNNProject/
│
├── Crack_Detection.py
│
├── CrackDataset/
│   ├── Positive/
│   └── Negative/
│
├── Processed_CrackDataset/          # Automatically created
│   ├── train/
│   │   ├── Crack/
│   │   └── NoCrack/
│   │
│   ├── validation/
│   │   ├── Crack/
│   │   └── NoCrack/
│   │
│   └── test/
│       ├── Crack/
│       └── NoCrack/
│
├── Best_Crack_Detection_Model.keras # Best model
├── Final_Crack_Detection_Model.keras # Final model
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `Processed_CrackDataset/` is automatically created by the Python program after running the project.

## ⚙️ Features

* Automatic dataset splitting
* Image resizing to `128 × 128`
* Data augmentation
* CNN-based image classification
* Binary classification:

  * Crack
  * No Crack
* Early Stopping
* Model Checkpoint
* Learning Rate Reduction
* Confusion Matrix
* Classification Report
* Single image prediction

## 🧠 CNN Architecture

```text
Input Image
    ↓
Conv2D
    ↓
BatchNormalization
    ↓
MaxPooling2D
    ↓
Conv2D
    ↓
BatchNormalization
    ↓
MaxPooling2D
    ↓
Conv2D
    ↓
BatchNormalization
    ↓
MaxPooling2D
    ↓
Conv2D
    ↓
BatchNormalization
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense
    ↓
Dropout
    ↓
Dense
    ↓
Dropout
    ↓
Output (Sigmoid)
```

## 📊 Dataset Split

The dataset is divided into:

* **70%** → Training
* **15%** → Validation
* **15%** → Testing

## 🚀 How to Run the Project

### Step 1: Create Virtual Environment

```cmd
python -m venv venv
```

### Step 2: Activate Virtual Environment

```cmd
venv\Scripts\activate
```

### Step 3: Install Required Libraries

```cmd
python -m pip install -r requirements.txt
```

### Step 4: Run the Project

```cmd
python Crack_Detection.py
```

## 💾 Model Files

### Best_Crack_Detection_Model.keras

Stores the model with the **best validation accuracy** during training.

### Final_Crack_Detection_Model.keras

Stores the final trained CNN model.

## 🔍 Prediction

The project can predict a single image and classify it as:

```text
Crack
```

or

```text
No Crack
```

## 👨‍💻 Author

**Yash Chavan**

## 📅 Date

**08/10/2026**
