import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_CSV  = "sign_dataset.csv"
MODEL_OUTPUT = "sign_model.h5"

MAX_SAMPLES  = 200   # hard cap per class  — prevents dominant classes overwhelming
MIN_SAMPLES  = 150   # hard floor per class — weak classes augmented up to this
JITTER_STD   = 0.015 # Gaussian noise std for augmentation

rng = np.random.default_rng(42)

df = pd.read_csv(DATASET_CSV)
X_raw = df.iloc[:, :63].values.astype(np.float32)
y_raw = df["label"].values

encoder = LabelEncoder()
y_enc_raw = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)
print(f"Classes ({num_classes}): {list(encoder.classes_)}\n")

# ── Hard-balance dataset ──────────────────────────────────────────────────────
balanced_X, balanced_y = [], []

for cls_idx in range(num_classes):
    mask    = y_enc_raw == cls_idx
    X_cls   = X_raw[mask]
    n       = len(X_cls)
    label   = encoder.classes_[cls_idx]

    if n > MAX_SAMPLES:
        # Downsample: random subset without replacement
        idx = rng.choice(n, MAX_SAMPLES, replace=False)
        X_cls = X_cls[idx]
        print(f"  CAPPED   '{label}': {n} → {MAX_SAMPLES}")
    elif n < MIN_SAMPLES:
        # Upsample: repeat + jitter until we hit MIN_SAMPLES
        needed = MIN_SAMPLES - n
        extra_idx = rng.choice(n, needed, replace=True)
        noise = rng.normal(0, JITTER_STD, (needed, 63)).astype(np.float32)
        X_extra = X_cls[extra_idx] + noise
        X_cls = np.vstack([X_cls, X_extra])
        print(f"  UPSAMPLED'{label}': {n} → {len(X_cls)}")
    else:
        print(f"  OK       '{label}': {n}")

    balanced_X.append(X_cls)
    balanced_y.append(np.full(len(X_cls), cls_idx, dtype=np.int32))

X_bal = np.concatenate(balanced_X)
y_bal = np.concatenate(balanced_y)

# Shuffle
idx = rng.permutation(len(X_bal))
X_bal, y_bal = X_bal[idx], y_bal[idx]
print(f"\nBalanced dataset: {X_bal.shape[0]} rows — {num_classes} classes\n")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# Light augmentation on train set only (1 extra jittered copy)
noise = rng.normal(0, JITTER_STD, X_train.shape).astype(np.float32)
X_train = np.vstack([X_train, X_train + noise])
y_train = np.concatenate([y_train, y_train])
shuf = rng.permutation(len(X_train))
X_train, y_train = X_train[shuf], y_train[shuf]

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat  = to_categorical(y_test,  num_classes=num_classes)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]

history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# Per-class accuracy
print("\n--- Per-class accuracy ---")
report = classification_report(y_test, y_pred, target_names=encoder.classes_, digits=2)
print(report)

# Confusion: show which classes are NEVER predicted
predicted_classes = set(np.unique(y_pred))
all_classes       = set(range(num_classes))
never_predicted   = all_classes - predicted_classes
if never_predicted:
    print("WARNING — classes never predicted:", [encoder.classes_[i] for i in sorted(never_predicted)])
else:
    print("✅ All classes predicted at least once.")

model.save(MODEL_OUTPUT)
print(f"\nModel saved to {MODEL_OUTPUT}")
np.save("sign_label_classes.npy", encoder.classes_)
print("Labels saved to sign_label_classes.npy")

# ── Training Graphs ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sign Language Landmark Model — Training History", fontsize=14, fontweight='bold')

epochs_ran = range(1, len(history.history['accuracy']) + 1)

axes[0].plot(epochs_ran, history.history['accuracy'],     label='Train Accuracy', marker='o', markersize=3)
axes[0].plot(epochs_ran, history.history['val_accuracy'], label='Val Accuracy',   marker='o', markersize=3)
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, history.history['loss'],     label='Train Loss', marker='o', markersize=3)
axes[1].plot(epochs_ran, history.history['val_loss'], label='Val Loss',   marker='o', markersize=3)
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
GRAPH_OUT = "sign_model_training_graph.png"
plt.savefig(GRAPH_OUT, dpi=150)
plt.close()
print(f"Training graph saved to {GRAPH_OUT}")
