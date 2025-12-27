# Skin Disease Classification using CNN and SVM

This repository contains an image classification system for skin diseases using
**Convolutional Neural Network (CNN)** and **Support Vector Machine (SVM)**.
The system is designed to classify skin condition images into three categories:
**Sehat**, **Panu**, and **Skabies**.

The CNN model is used as a deep learning approach for end-to-end image classification,
while the SVM model serves as a traditional machine learning baseline using flattened
grayscale image features.

---

## 📁 Project Structure

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
│   └── image_visualizations (*.png)
│
├── venv/            # virtual environment (ignored)
├── .gitignore
└── README.md

---

## 🧠 Models Implemented

### 1. Convolutional Neural Network (CNN)
- Input image size: **50 × 50**
- Image mode: **Grayscale**
- Architecture:
  - Convolution + Max Pooling layers
  - Fully Connected layer with Dropout
  - Softmax output layer (3 classes)
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**

### 2. Support Vector Machine (SVM)
- Kernel: **Linear**
- Input: Flattened grayscale image vectors
- Feature normalization: Pixel values scaled to range 0–1

---

## ⚙️ Requirements

- Python **3.10 / 3.11**
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies:

pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow

---

## ▶️ How to Run

1. Prepare the dataset following the folder structure in `data/train` and `data/test`.
2. Activate the virtual environment (optional).
3. Run the model:

cd models  
python model_cnn_svm.py

---

## 📊 Output

The program will automatically generate and save the following outputs:

- CNN training accuracy and loss curves  
- CNN confusion matrix  
- SVM confusion matrix  
- Sample image visualizations for each class  

All outputs are stored in the `results/` directory.

---

## 📌 Important Notes on Accuracy

The accuracy of the CNN and SVM models is **highly dependent on the size and quality of the dataset**.
To obtain more reliable and generalizable results, it is strongly recommended to:

- Use a **large number of training images** for each class
- Ensure **balanced data distribution** across classes
- Use images with **consistent quality and resolution**
- Consider applying **data augmentation** techniques
- Perform **cross-validation** for more robust evaluation

Using a limited dataset may result in overfitting and reduced model performance on unseen data.

---

## 🔬 Research and Academic Use

This project can be used as:

- A baseline experiment for medical image classification
- A comparison study between CNN and SVM approaches
- Supporting material for academic research, thesis, or journal publications

---

## 👤 Author

Developed by **Hardika Setiyawan**  
Informatics – Intelligent Systems / Machine Learning
