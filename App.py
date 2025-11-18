from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import os
import numpy as np
import json
import time

app = Flask(__name__)

# ----------------- Config -----------------
CRIMINAL_DIR = "criminal_data"      # each person: criminal_data/<name>/*.jpg
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.json"
os.makedirs(CRIMINAL_DIR, exist_ok=True)

# Haar cascades (bundled with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Create LBPH recognizer (requires opencv-contrib)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    recognizer = None
    print("Warning: OpenCV-contrib is required for LBPH recognizer (cv2.face).", e)


# ----------------- Helper: train model from criminal_data -----------------
def train_model():
    if recognizer is None:
        return {"status": "error", "message": "LBPH recognizer not available (opencv-contrib required)."}

    label_ids = {}
    current_id = 0
    x_train = []
    y_labels = []

    for person in os.listdir(CRIMINAL_DIR):
        p_path = os.path.join(CRIMINAL_DIR, person)
        if not os.path.isdir(p_path):
            continue

        # Ensure label mapping
        if person not in label_ids:
            label_ids[person] = current_id
            current_id += 1

        for file in os.listdir(p_path):
            fpath = os.path.join(p_path, file)
            if not fpath.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # detect face and crop (to improve training)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                # if no face detected, resize whole image
                face_img = cv2.resize(img, (200, 200))
                x_train.append(face_img)
                y_labels.append(label_ids[person])
            else:
                for (x, y, w, h) in faces:
                    face_roi = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    x_train.append(face_resized)
                    y_labels.append(label_ids[person])

    if len(x_train) == 0:
        return {"status": "error", "message": "No training images found. Add criminal images first."}

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_ids, f)

    return {"status": "ok", "labels": label_ids, "trained_images": len(x_train)}


# ----------------- Helper: load labels -----------------
def load_labels():
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, "r") as f:
        mapping = json.load(f)
    # invert mapping to id->name
    inv = {int(v): k for k, v in mapping.items()}
    return inv


# ----------------- Routes: UI -----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/criminal")
def criminal_page():
    # display existing criminals
    people = [d for d in os.listdir(CRIMINAL_DIR) if os.path.isdir(os.path.join(CRIMINAL_DIR, d))]
    return render_template("criminal.html", people=people)


@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html")


# ----------------- Add criminal (upload single or multiple) -----------------
@app.route("/add_criminal", methods=["POST"])
def add_criminal():
    name = request.form.get("name", "").strip()
    files = request.files.getlist("images")

    if not name:
        return "Name required", 400
    if len(files) == 0 or files[0].filename == "":
        return "At least one image required", 400

    person_dir = os.path.join(CRIMINAL_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    saved = 0
    for f in files:
        filename = f.filename
        save_path = os.path.join(person_dir, filename)
        f.save(save_path)
        saved += 1

    return redirect(url_for("criminal_page"))


# ----------------- Trigger training -----------------
@app.route("/train", methods=["POST"])
def train_route():
    result = train_model()
    return jsonify(result)


# ----------------- Simple health route to check model loaded -----------------
@app.route("/model_status")
def model_status():
    status = {"recognizer": bool(recognizer), "model_file": os.path.exists(MODEL_PATH)}
    return jsonify(status)


# ----------------- Video stream generators -----------------
def gen_criminal_frames():
    # Criminal detection stream: use LBPH to predict
    cap = cv2.VideoCapture(0)
    frame_id = 0
    labels = load_labels()

    # load model if exists
    if recognizer is not None and os.path.exists(MODEL_PATH):
        recognizer.read(MODEL_PATH)
    else:
        # If no model, still stream but always "Unknown"
        print("Model not loaded; predictions will not run.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to reduce CPU
        small = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        display_frame = small.copy()

        frame_id += 1

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))

            name_label = "Unknown"
            conf_label = None

            # run prediction intermittently (every 4 frames)
            if recognizer is not None and os.path.exists(MODEL_PATH) and frame_id % 4 == 0:
                try:
                    label_id, confidence = recognizer.predict(face_resized)  # lower confidence = better match
                    conf_label = confidence
                    # threshold: 80 typical, adjust if necessary
                    if confidence < 80:
                        labels_map = load_labels()
                        name_label = labels_map.get(label_id, "Unknown")
                    else:
                        name_label = "Unknown"
                except Exception:
                    name_label = "Unknown"

            # draw box and label (red for criminal match)
            if name_label != "Unknown":
                color = (0, 0, 255)  # red
                text = f"CRIMINAL: {name_label} ({int(conf_label)})"
            else:
                color = (0, 200, 0)  # green
                text = "Innocent / Unknown"

            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def gen_emotion_frames():
    # Lightweight emotion heuristic using Haar cascades (smile + eyes)
    cap = cv2.VideoCapture(0)
    frame_id = 0
    last_label = "Normal Person"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        display_frame = small.copy()
        frame_id += 1

        for (x, y, w, h) in faces:
            roi_color = display_frame[y:y+h, x:x+w]
            roi_gray  = gray[y:y+h, x:x+w]

            # detect smile and eyes (very fast)
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)

            # Simple Option A mapping:
            # - If smile found -> Normal Person
            # - Else if smile not found and eyes found -> Pocket Cutter Suspect
            # - Else if neither smile nor eyes detected -> Psycho (suspicious)
            # - default -> Normal Person
            if len(smiles) > 0:
                label = "Normal Person"
                color = (0, 255, 0)
            elif len(eyes) > 0 and len(smiles) == 0:
                label = "Pocket Cutter Suspect"
                color = (0, 140, 255)  # orange
            elif len(eyes) == 0 and len(smiles) == 0:
                # if faces visible but no eyes/smile detected -> label as Psycho (heuristic)
                label = "Psycho / Dangerous (Heuristic)"
                color = (0, 0, 255)
            else:
                label = "Normal Person"
                color = (0, 255, 0)

            # draw rectangle + label
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        # tiny sleep to keep CPU friendly
        time.sleep(0.02)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ----------------- Routes for video feeds -----------------
@app.route("/video_feed_criminal")
def video_feed_criminal():
    return Response(gen_criminal_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/video_feed_emotion")
def video_feed_emotion():
    return Response(gen_emotion_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------- Utility: delete person (optional) -----------------
@app.route("/delete_person", methods=["POST"])
def delete_person():
    name = request.form.get("name", "")
    if not name:
        return jsonify({"status":"error","msg":"name required"}), 400
    path = os.path.join(CRIMINAL_DIR, name)
    if os.path.exists(path):
        # remove files and folder
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.rmdir(path)
        return redirect(url_for("criminal_page"))
    return jsonify({"status":"error","msg":"not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
