import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "sign_model.h5"
LABELS_PATH = "sign_label_classes.npy"

model = load_model(MODEL_PATH)
classes = np.load(LABELS_PATH, allow_pickle=True)

print(f"Loaded model. Classes: {list(classes)}\n")

def predict(landmarks_63):
    x = np.array(landmarks_63, dtype=np.float32).reshape(1, 63)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argmax(probs)
    return classes[idx], round(float(probs[idx]) * 100, 2)

# --- Test with a random sample (replace with real landmarks) ---
sample = np.random.rand(63).tolist()
word, confidence = predict(sample)
print(f"Predicted: {word}  ({confidence}% confidence)")
