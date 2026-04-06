import os
import csv
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VIDEOS_FOLDER = "sign_videos"
OUTPUT_CSV    = "sign_dataset.csv"
MODEL_PATH    = "hand_landmarker.task"

# Words that are MOTION-based and cannot be reliably captured from single frames.
SKIP_LABELS = {"breathe", "untitled folder"}

# Minimum frames before a label is accepted into the CSV.
MIN_FRAMES_PER_LABEL = 30

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm")


def normalize_landmarks(landmarks):
    """
    Translate all landmarks so wrist (index 0) is at origin,
    then scale by palm size (wrist → middle-finger MCP = index 9).
    Returns a flat list of 63 floats, or None if the hand is degenerate.
    """
    wx, wy, wz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    # Palm size: Euclidean distance wrist → middle MCP (landmark 9)
    palm_size = math.sqrt(
        (landmarks[9].x - wx) ** 2 +
        (landmarks[9].y - wy) ** 2
    )
    if palm_size < 1e-5:
        return None  # degenerate / hand too far / occluded
    features = []
    for lm in landmarks:
        features.append((lm.x - wx) / palm_size)
        features.append((lm.y - wy) / palm_size)
        features.append((lm.z - wz) / palm_size)
    return features


base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

# --- Build a unified map: label → [list of video file paths] ---
# Pass 1: subfolders  →  label = folder name, all videos inside
# Pass 2: flat files  →  label = filename without extension
label_videos = {}

for entry in sorted(os.scandir(VIDEOS_FOLDER), key=lambda e: e.name.lower()):
    lbl = entry.name.lower().strip()
    if lbl in SKIP_LABELS:
        continue
    if entry.is_dir():
        videos = sorted(
            os.path.join(entry.path, f)
            for f in os.listdir(entry.path)
            if f.lower().endswith(VIDEO_EXTS)
        )
        if videos:
            label_videos.setdefault(lbl, []).extend(videos)
    elif entry.is_file() and entry.name.lower().endswith(VIDEO_EXTS):
        lbl = os.path.splitext(lbl)[0]
        if lbl not in SKIP_LABELS:
            label_videos.setdefault(lbl, []).append(entry.path)

print(f"Found {len(label_videos)} labels across subfolder + flat-file sources.\n")

# --- Extract landmarks for every label ---
label_rows = {}

for label, video_paths in sorted(label_videos.items()):
    seen_vectors = set()   # dedup across ALL videos for this label
    all_rows     = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        vid_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            features = normalize_landmarks(result.hand_landmarks[0])
            if features is None:
                continue

            fingerprint = tuple(round(v, 4) for v in features)
            if fingerprint in seen_vectors:
                continue
            seen_vectors.add(fingerprint)

            all_rows.append(features + [label])
            vid_count += 1

        cap.release()

    total = len(all_rows)
    print(f"  '{label}': {len(video_paths)} video(s) → {total} unique frames")
    label_rows[label] = all_rows

detector.close()

# Write only labels that pass the minimum threshold
print("\n--- Writing dataset ---")
label_counts = {}
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for label, rows in sorted(label_rows.items()):
        if len(rows) < MIN_FRAMES_PER_LABEL:
            print(f"  DROPPED '{label}': only {len(rows)} frames (< {MIN_FRAMES_PER_LABEL})")
            continue
        for row in rows:
            writer.writerow(row)
        label_counts[label] = len(rows)
        print(f"  KEPT   '{label}': {len(rows)} frames")

print(f"\nDataset saved to {OUTPUT_CSV}")
print(f"Total rows: {sum(label_counts.values())}  |  Classes: {len(label_counts)}")

low = {k: v for k, v in label_counts.items() if v < 80}
if low:
    print(f"\nWARNING — labels with < 80 samples (retrain needed): {low}")
