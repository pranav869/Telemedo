"""
plot_model_graphs.py
Generates evaluation graphs for all trained models.
Does NOT modify any existing code — run independently.
Outputs PNG files in the project root.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 1 — Sign Language Landmark Model  (sign_model.h5)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL 1: Sign Language Landmark Model (sign_model.h5)")
print("=" * 60)

MODEL_PATH   = "sign_model.h5"
DATASET_CSV  = "sign_dataset.csv"
LABELS_PATH  = "sign_label_classes.npy"
MAX_SAMPLES  = 200
MIN_SAMPLES  = 150
JITTER_STD   = 0.015

rng = np.random.default_rng(42)

df    = pd.read_csv(DATASET_CSV)
X_raw = df.iloc[:, :63].values.astype(np.float32)
y_raw = df["label"].values

encoder    = LabelEncoder()
y_enc_raw  = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)
print(f"Classes ({num_classes}): {list(encoder.classes_)}\n")

# ── Replicate same hard-balance as train_model.py ────────────────────────────
balanced_X, balanced_y = [], []
class_counts_before, class_counts_after = [], []

for cls_idx in range(num_classes):
    mask   = y_enc_raw == cls_idx
    X_cls  = X_raw[mask]
    n      = len(X_cls)
    class_counts_before.append(n)

    if n > MAX_SAMPLES:
        idx   = rng.choice(n, MAX_SAMPLES, replace=False)
        X_cls = X_cls[idx]
    elif n < MIN_SAMPLES:
        needed    = MIN_SAMPLES - n
        extra_idx = rng.choice(n, needed, replace=True)
        noise     = rng.normal(0, JITTER_STD, (needed, 63)).astype(np.float32)
        X_cls     = np.vstack([X_cls, X_cls[extra_idx] + noise])

    class_counts_after.append(len(X_cls))
    balanced_X.append(X_cls)
    balanced_y.append(np.full(len(X_cls), cls_idx, dtype=np.int32))

X_bal = np.concatenate(balanced_X)
y_bal = np.concatenate(balanced_y)
idx   = rng.permutation(len(X_bal))
X_bal, y_bal = X_bal[idx], y_bal[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ── Load model & predict ──────────────────────────────────────────────────────
model  = load_model(MODEL_PATH)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

report = classification_report(
    y_test, y_pred, target_names=encoder.classes_, output_dict=True
)
per_class_f1       = [report[c]['f1-score']  for c in encoder.classes_]
per_class_precision= [report[c]['precision'] for c in encoder.classes_]
per_class_recall   = [report[c]['recall']    for c in encoder.classes_]

print(classification_report(y_test, y_pred, target_names=encoder.classes_, digits=2))

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 1 — Class Distribution (before vs after balancing)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, num_classes * 0.45), 5))
x = np.arange(num_classes)
w = 0.4
ax.bar(x - w/2, class_counts_before, width=w, label='Before Balancing', color='#6366f1', alpha=0.85)
ax.bar(x + w/2, class_counts_after,  width=w, label='After Balancing',  color='#22c55e', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(encoder.classes_, rotation=90, fontsize=8)
ax.set_title("Sign Language Model — Class Distribution (Before vs After Balancing)",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Sample Count")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("graph_sign_class_distribution.png", dpi=150)
plt.close()
print("✅ Saved: graph_sign_class_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 2 — Per-Class F1 / Precision / Recall
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, num_classes * 0.45), 5))
x = np.arange(num_classes)
w = 0.28
ax.bar(x - w,   per_class_f1,        width=w, label='F1-Score',  color='#3b82f6', alpha=0.9)
ax.bar(x,       per_class_precision, width=w, label='Precision', color='#f59e0b', alpha=0.9)
ax.bar(x + w,   per_class_recall,    width=w, label='Recall',    color='#ef4444', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(encoder.classes_, rotation=90, fontsize=8)
ax.set_ylim(0, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
ax.set_title("Sign Language Model — Per-Class F1 / Precision / Recall",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Score")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("graph_sign_per_class_metrics.png", dpi=150)
plt.close()
print("✅ Saved: graph_sign_per_class_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 3 — Confusion Matrix  (top-N classes if too many)
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

if num_classes <= 30:
    fig_size = (max(12, num_classes * 0.55), max(10, num_classes * 0.55))
    fig, ax = plt.subplots(figsize=fig_size)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=90, cmap='Blues')
    ax.set_title("Sign Language Model — Confusion Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("graph_sign_confusion_matrix.png", dpi=120)
    plt.close()
    print("✅ Saved: graph_sign_confusion_matrix.png")
else:
    # Too many classes — show heatmap without labels
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Sign Language Model — Confusion Matrix ({num_classes} classes)\n"
                 "Rows = True, Cols = Predicted", fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Class Index")
    ax.set_ylabel("True Class Index")
    plt.tight_layout()
    plt.savefig("graph_sign_confusion_matrix.png", dpi=120)
    plt.close()
    print("✅ Saved: graph_sign_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 4 — Overall Accuracy / Macro Avg Summary Card
# ─────────────────────────────────────────────────────────────────────────────
overall_acc   = report['accuracy']
macro_f1      = report['macro avg']['f1-score']
macro_prec    = report['macro avg']['precision']
macro_recall  = report['macro avg']['recall']

metrics_names  = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
metrics_values = [overall_acc, macro_f1, macro_prec, macro_recall]
colors = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(metrics_names, metrics_values, color=colors, alpha=0.88, height=0.5)
ax.set_xlim(0, 1.15)
for bar, val in zip(bars, metrics_values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%", va='center', fontsize=11, fontweight='bold')
ax.set_title("Sign Language Landmark Model — Overall Performance",
             fontsize=13, fontweight='bold')
ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("graph_sign_overall_metrics.png", dpi=150)
plt.close()
print("✅ Saved: graph_sign_overall_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All graphs generated successfully!")
print("  Files:")
print("    graph_sign_class_distribution.png")
print("    graph_sign_per_class_metrics.png")
print("    graph_sign_confusion_matrix.png")
print("    graph_sign_overall_metrics.png")
print("=" * 60)
