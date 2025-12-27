# model_cnn_svm_final.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- PATH ---
train_dir = '../data/train'
test_dir = '../data/test'

# --- PARAMETER ---
img_size = 50
batch_size = 16
epochs = 10

# --- FLAG: MAU GRAYSCALE ATAU RGB ---
USE_GRAYSCALE = True

# --- DATA GENERATOR ---
datagen = ImageDataGenerator(rescale=1./255)

color_mode = 'grayscale' if USE_GRAYSCALE else 'rgb'

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode=color_mode,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode=color_mode,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --- CNN MODEL ---
input_shape = (img_size, img_size, 1) if USE_GRAYSCALE else (img_size, img_size, 3)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN CNN ---
history = cnn_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# --- EVALUATE CNN ---
test_generator.reset()
cnn_preds = cnn_model.predict(test_generator)
cnn_pred_labels = np.argmax(cnn_preds, axis=1)
true_labels = test_generator.classes

# --- CONFUSION MATRIX CNN ---
cm_cnn = confusion_matrix(true_labels, cnn_pred_labels)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Purples' if USE_GRAYSCALE else 'Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
title_cnn = 'CNN - Confusion Matrix (Grayscale)' if USE_GRAYSCALE else 'CNN - Confusion Matrix (RGB)'
plt.title(title_cnn)
plt.savefig('../results/cnn_confusion_matrix_final.png')
plt.show()

# --- ACCURACY & LOSS ---
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('CNN - Accuracy (Grayscale)' if USE_GRAYSCALE else 'CNN - Accuracy (RGB)')
plt.legend()
plt.savefig('../results/cnn_accuracy_curve_final.png')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('CNN - Loss (Grayscale)' if USE_GRAYSCALE else 'CNN - Loss (RGB)')
plt.legend()
plt.savefig('../results/cnn_loss_curve_final.png')
plt.show()

# --- SVM PREP ---
def load_data_for_svm(data_dir):
    X = []
    y = []
    class_names = os.listdir(data_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y)

X_train_svm, y_train_svm = load_data_for_svm(train_dir)
X_test_svm, y_test_svm = load_data_for_svm(test_dir)

X_train_svm = X_train_svm / 255.0
X_test_svm = X_test_svm / 255.0

# --- SVM MODEL ---
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_svm, y_train_svm)

# --- PREDICT SVM ---
svm_preds = svm_model.predict(X_test_svm)

# --- CONFUSION MATRIX SVM ---
cm_svm = confusion_matrix(y_test_svm, svm_preds)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('SVM - Confusion Matrix (Grayscale)')
plt.savefig('../results/svm_confusion_matrix_final.png')
plt.show()

# --- CLASSIFICATION REPORT ---
print("\n--- CNN Classification Report ---")
print(classification_report(true_labels, cnn_pred_labels, target_names=test_generator.class_indices.keys()))

print("\n--- SVM Classification Report ---")
print(classification_report(y_test_svm, svm_preds, target_names=test_generator.class_indices.keys()))

# --- GRID GAMBAR PILIH MANUAL (Grayscale) ---
classes = ['sehat', 'panu', 'skabies']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

selected_images = {
    'sehat': ['thumb (41).jpg', 'thumb (46).jpg', 'thumb (13).jpg'],
    'panu': ['13candidaTongue082205.jpg', 'actionomycosis-3.jpg', 'id-reaction-6.jpg'],
    'skabies': ['scabies-96.jpg', 'jelly-fish-sting7.jpg', 'cactus-granuloma-2.jpg']
}

for i, cls in enumerate(classes):
    class_folder = f'../data/train/{cls}'
    img_files = selected_images[cls]
    
    for j, img_file in enumerate(img_files):
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{cls} - {img_file}')
            ax.axis('off')
        else:
            print(f"WARNING: File not found: {img_path}")

plt.suptitle('Contoh Gambar Penyakit Kulit (Pilihan Manual - Grayscale)', fontsize=16)
plt.tight_layout()
plt.savefig('../results/contoh_gambar_pilihan_manual_grayscale_final.png')
plt.show()

# --- SEMUA GAMBAR PER KELAS ---
for cls in classes:
    class_folder = f'../data/train/{cls}'
    img_files = os.listdir(class_folder)
    n_images = len(img_files)
    ncols = 5
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()
    
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{cls}')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Semua Gambar Kelas {cls} (Grayscale)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../results/semua_gambar_{cls}_grayscale_final.png')
    plt.show()
    