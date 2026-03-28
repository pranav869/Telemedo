import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- CONFIG ---
DATASET_DIR = 'assets/dataset/SL'
IMG_SIZE = 96  # Resize all frames to 96x96
FRAMES_PER_VIDEO = 1  # Use 1 frame per video for simple classification
BATCH_SIZE = 32
EPOCHS = 10

# --- DATA LOADING ---
def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame

def load_data():
    X, y, class_names = [], [], []
    class_folders = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    class_names = class_folders
    for idx, label in enumerate(class_folders):
        video_files = glob(os.path.join(DATASET_DIR, label, '*.mp4'))
        for video in tqdm(video_files, desc=f'Loading {label}'):
            frame = extract_frame(video)
            if frame is not None:
                X.append(frame)
                y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

print('Loading data...')
X, y, class_names = load_data()
print(f'Total samples: {len(X)}, Classes: {len(class_names)}')

# --- PREPROCESS ---
X = X.astype('float32') / 255.0
y_cat = to_categorical(y, num_classes=len(class_names))

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.05, random_state=42, stratify=None)

# --- MODEL ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# --- TRAIN ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# --- SAVE ---
model.save('sign_to_text_cnn.h5')
with open('sign_to_text_labels.txt', 'w') as f:
    for label in class_names:
        f.write(label + '\n')
print('Model and labels saved.')
