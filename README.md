
---

```markdown
# Skin Disease Classification using CNN and SVM

This repository presents an image classification system for skin disease detection using
**Convolutional Neural Network (CNN)** and **Support Vector Machine (SVM)**.
The system classifies skin condition images into three categories:
**Sehat**, **Panu**, and **Skabies**.

CNN is used as a deep learning approach for end-to-end image classification,
while SVM serves as a traditional machine learning baseline using flattened
grayscale image features.

---

## 📁 Project Structure

```

penyakit_projek_kulit/
│
├── data/
│   ├── train/
│   │   ├── sehat/
│   │   ├── panu/
│   │   └── skabies/
│   │
│   └── test/
│       ├── sehat/
│       ├── panu/
│       └── skabies/
│
├── models/
│   └── model_cnn_svm.py
│
├── results/
│   ├── cnn_accuracy_curve_final.png
│   ├── cnn_loss_curve_final.png
│   ├── cnn_confusion_matrix_final.png
│   ├── svm_confusion_matrix_final.png
│   └── visualization_outputs.png
│
├── venv/            # virtual environment (ignored)
├── .gitignore
└── README.md

```

> **Note:** The `data`, `results`, and `venv` directories are excluded from version control.

---

## 🧠 Models Implemented

### Convolutional Neural Network (CNN)
- Input image size: **50 × 50**
- Image type: **Grayscale**
- Architecture:
  - Convolutional layers with Max Pooling
  - Fully connected layer with Dropout
  - Softmax output layer (3 classes)
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**

### Support Vector Machine (SVM)
- Kernel: **Linear**
- Input features: Flattened grayscale image vectors
- Feature scaling: Pixel values normalized to **0–1**

---

## ⚙️ Requirements

- Python **3.10** or **3.11**
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies:

```

pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow

```

---

## ▶️ How to Run

1. Prepare the dataset according to the folder structure in `data/train` and `data/test`.
2. (Optional) Activate the virtual environment.
3. Run the training and evaluation script:

```

cd models
python model_cnn_svm.py

```

---

## 📊 Output

After execution, the system automatically generates:

- CNN training accuracy and loss curves
- CNN confusion matrix
- SVM confusion matrix
- Sample image visualizations for each class

All output files are saved in the `results/` directory.

---

## 📌 Notes on Model Accuracy

The accuracy of both CNN and SVM models is **highly dependent on the quantity and quality of the dataset**.
For more reliable and generalizable results, it is strongly recommended to:

- Use a **large number of training images** for each class
- Ensure **balanced data distribution**
- Maintain **consistent image quality and resolution**
- Apply **data augmentation** techniques
- Perform **cross-validation** for robust evaluation

Using a limited dataset may lead to overfitting and reduced performance on unseen data.

---

## 🔬 Research and Academic Use

This project can be used for:
- Baseline experiments in medical image classification
- Comparative studies between deep learning and traditional machine learning methods
- Supporting material for academic research, thesis, or journal publications

---

## 👤 Author

**Hardika Setiyawan**  
Informatics – Intelligent Systems / Machine Learning
```