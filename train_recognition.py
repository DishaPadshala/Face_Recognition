import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, losses, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = 'faces_dataset'
TRAINING_DIR = 'data/train'
TESTING_DIR  = 'data/test'
MODEL_PATH   = 'face_recognition_model.keras'
LE_PATH      = 'label_encoder.pickle'
IMAGE_HEIGHT = 160
IMAGE_WIDTH  = 160
BATCH_SIZE   = 20
EPOCHS       = 100
TEST_SPLIT   = 0.2

# ── MCDropout ─────────────────────────────────────────────────────────────────
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# ── Load face directly (already cropped) ─────────────────────────────────────
def extract_face(image_path, required_size=(160, 160)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image     = cv2.resize(image, required_size)
        gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_face
    except Exception as e:
        print(f"[WARN] Error processing {image_path}: {e}")
        return None

# ── Save dataset in Keras directory structure ─────────────────────────────────
def save_dataset(setname, data, labels):
    for label, gray_img in tqdm(zip(labels, data)):
        directory = f"data/{setname}/{label}/"
        os.makedirs(directory, exist_ok=True)
        existing = len(os.listdir(directory))
        cv2.imwrite(f"{directory}{label}_{existing+1}.png", gray_img)

# ── Load and split dataset ────────────────────────────────────────────────────
print("[INFO] Loading images...")
shutil.rmtree('data', ignore_errors=True)

all_faces  = []
all_labels = []

for person_name in sorted(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    print(f"[INFO] Processing {person_name}...")
    count = 0
    for img_file in tqdm(os.listdir(person_dir)):
        img_path = os.path.join(person_dir, img_file)
        face = extract_face(img_path)
        if face is not None:
            all_faces.append(face)
            all_labels.append(person_name)
            count += 1
    print(f"  → {count} faces loaded")

print(f"[INFO] Total: {len(all_faces)} faces, {len(set(all_labels))} classes")

# ── Split into train/test ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    all_faces, all_labels,
    test_size=TEST_SPLIT,
    random_state=42,
    stratify=all_labels
)
print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# ── Save to directory structure ───────────────────────────────────────────────
print("[INFO] Saving dataset...")
save_dataset("train", X_train, y_train)
save_dataset("test",  X_test,  y_test)

# ── Save label encoder ────────────────────────────────────────────────────────
classes     = sorted(list(set(all_labels)))
NUM_CLASSES = len(classes)
print(f"[INFO] Classes ({NUM_CLASSES}): {classes}")

with open(LE_PATH, 'wb') as f:
    pickle.dump(classes, f)

# ── Data generators ───────────────────────────────────────────────────────────
training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
).flow_from_directory(
    TRAINING_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'
)

testing_generator = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    TESTING_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    TESTING_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

NUM_TRAINING = len(X_train)
NUM_TESTING  = len(X_test)

# ── CNN Model ─────────────────────────────────────────────────────────────────
print("[INFO] Building model...")
model = models.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu',
                kernel_initializer="glorot_uniform",
                kernel_regularizer=regularizers.l2(l2=0.01)))

model.add(MCDropout(rate=0.5))

model.add(Dense(NUM_CLASSES, activation='softmax',
                kernel_initializer="glorot_uniform"))

model.summary()

model.compile(
    loss=losses.CategoricalCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.0003),
    metrics=["accuracy"]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("[INFO] Training...")
history = model.fit(
    training_generator,
    steps_per_epoch=NUM_TRAINING // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=testing_generator,
    validation_steps=NUM_TESTING // BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("[INFO] Evaluating...")
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

print(classification_report(
    validation_generator.classes, y_pred,
    target_names=list(validation_generator.class_indices.keys()),
    zero_division=0
))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=list(validation_generator.class_indices.keys()),
            yticklabels=list(validation_generator.class_indices.keys()))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("[INFO] Confusion matrix saved to confusion_matrix.png")

# ── Training curves ───────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("[INFO] Training curves saved to training_curves.png")
print(f"[INFO] Model saved to {MODEL_PATH}")