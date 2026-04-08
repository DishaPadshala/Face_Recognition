from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import imutils
import pickle
import cv2
import os
import keras

# ── MCDropout ─────────────────────────────────────────────────────────────────
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH           = 'face_recognition_model.keras'
LE_PATH              = 'label_encoder.pickle'
CONFIDENCE_THRESHOLD = 0.7
IMG_SIZE             = 160
SIDEBAR_WIDTH        = 300

# ── Load model and label encoder ─────────────────────────────────────────────
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, custom_objects={'MCDropout': MCDropout}, compile=False)

print("[INFO] Loading label encoder...")
with open(LE_PATH, 'rb') as f:
    classes = pickle.load(f)

print(f"[INFO] Classes: {classes}")

# ── Face detector ─────────────────────────────────────────────────────────────
print("[INFO] Loading face detector...")
detector = MTCNN()

# ── Attendance ────────────────────────────────────────────────────────────────
def markAttendance(name):
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Date,Time\n')
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList   = [line.split(',')[0] for line in myDataList]
        now        = datetime.now()
        date_str   = now.strftime('%Y-%m-%d')
        time_str   = now.strftime('%H:%M:%S')
        if name not in nameList:
            f.write(f'{name},{date_str},{time_str}\n')

# ── Draw sidebar ──────────────────────────────────────────────────────────────
def draw_sidebar(frame, current_name, current_conf):
    h       = frame.shape[0]
    sidebar = np.zeros((h, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    sidebar[:] = (30, 30, 30)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(sidebar, "FACE RECOGNITION", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(sidebar, (10, 45), (SIDEBAR_WIDTH - 10, 45), (100, 100, 100), 1)

    # ── Current detection ─────────────────────────────────────────────────────
    cv2.putText(sidebar, "DETECTED", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    color = (0, 255, 0) if current_name not in ["Unknown", "No Face"] else (0, 0, 255)
    cv2.putText(sidebar, current_name, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if current_name not in ["Unknown", "No Face"]:
        cv2.putText(sidebar, f"Confidence: {current_conf*100:.1f}%", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # confidence bar
        bar_width = int((SIDEBAR_WIDTH - 20) * current_conf)
        cv2.rectangle(sidebar, (10, 140), (SIDEBAR_WIDTH - 10, 155), (80, 80, 80), -1)
        cv2.rectangle(sidebar, (10, 140), (10 + bar_width, 155), (0, 255, 0), -1)

    cv2.line(sidebar, (10, 170), (SIDEBAR_WIDTH - 10, 170), (100, 100, 100), 1)
    cv2.line(sidebar, (10, h - 50), (SIDEBAR_WIDTH - 10, h - 50), (100, 100, 100), 1)

    # ── Footer ────────────────────────────────────────────────────────────────
    now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    cv2.putText(sidebar, now, (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(sidebar, "Press Q to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return sidebar

# ── Start webcam ──────────────────────────────────────────────────────────────
print("[INFO] Starting webcam...")
vs = cv2.VideoCapture(0)

if not vs.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

current_name = "No Face"
current_conf = 0.0

while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        print("[ERROR] Cannot read from camera.")
        break

    frame = imutils.resize(frame, width=600)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Detect faces ──────────────────────────────────────────────────────────
    try:
        results = detector.detect_faces(rgb)
    except Exception:
        results = []

    if len(results) == 0:
        current_name = "No Face"
        current_conf = 0.0

    for result in results:
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)
        x2, y2 = x + w, y + h
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face = rgb[y:y2, x:x2]
        if face.size == 0:
            continue

        # ── Preprocess ────────────────────────────────────────────────────────
        face_gray    = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
        face_norm    = face_resized.astype("float") / 255.0
        face_input   = np.expand_dims(face_norm, axis=-1)
        face_input   = np.expand_dims(face_input, axis=0)

        # ── Predict ───────────────────────────────────────────────────────────
        preds        = model.predict(face_input, verbose=0)[0]
        current_conf = np.max(preds)
        class_idx    = np.argmax(preds)

        if current_conf >= CONFIDENCE_THRESHOLD:
            current_name = classes[class_idx]
            color        = (0, 255, 0)
            markAttendance(current_name)
        else:
            current_name = "Unknown"
            color        = (0, 0, 255)

        # ── Draw on frame ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        label   = f"{current_name} ({current_conf*100:.1f}%)"
        label_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(frame, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Combine frame and sidebar ─────────────────────────────────────────────
    sidebar  = draw_sidebar(frame, current_name, current_conf)
    combined = np.hstack((frame, sidebar))

    cv2.imshow("Face Recognition System", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()