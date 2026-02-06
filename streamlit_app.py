import streamlit as st
import cv2
import tempfile
from utils import features_to_extract, count_pullup,count_pushup,count_situp, rescale_frame
from ultralytics import YOLO
import numpy as np
from xgboost import XGBClassifier
import pickle
import os

st.title("Exercise Tracker")
st.write(
    "Upload either pull up, push up and sit up and this will identify it."
)
uploaded_file = st.file_uploader(
    "Upload video please",
    type = ["mp4"]
)
@st.cache_resource
def load_models():
    # Load XGB model
    with open("exercise_identifier_xgb.pkl", "rb") as f:
        xgb = pickle.load(f)

    # Load YOLO safely
    yolo_path = os.path.join(os.getcwd(), "yolo26n-pose.pt")
    yolo = YOLO(yolo_path)

    return xgb, yolo

loaded_xgb, model = load_models()

states = {"push_up": "UP", "pull_up": "UP", "sit_up": "UP"}
counters = {"push_up": 0, "pull_up": 0, "sit_up": 0}

buffer = [] #require to collect features for a window
window_size = 30
prediction = 'None Detected'

if uploaded_file:

    st.write("Processing video...")

    # Save input video
    input_tfile = tempfile.NamedTemporaryFile(delete=False)
    input_tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(input_tfile.name)

    # Output temp file
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Video writer setup
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # State vars
    states = {"push_up": "UP", "pull_up": "UP", "sit_up": "UP"}
    counters = {"push_up": 0, "pull_up": 0, "sit_up": 0}

    buffer = []
    window_size = 30
    prediction = "Detecting..."

    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # YOLO
        results = model(frame, conf=0.2)
        r = results[0]

        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()

            h_, w_ = frame.shape[:2]
            kpts[:, 0] /= w_
            kpts[:, 1] /= h_

            feat = features_to_extract(kpts)
            buffer.append(feat)

            if len(buffer) > window_size:
                buffer.pop(0)

            if len(buffer) == window_size:
                X_input = np.array(buffer).reshape(1, -1)
                prediction = loaded_xgb.predict(X_input)[0]

                if prediction == 1:
                    angle = buffer[-1][1]
                    states["push_up"], counters["push_up"] = count_pushup(
                        angle, states["push_up"], counters["push_up"]
                    )

                elif prediction == 0:
                    angle = buffer[-1][1]
                    states["pull_up"], counters["pull_up"] = count_pullup(
                        angle, states["pull_up"], counters["pull_up"]
                    )

                elif prediction == 2:
                    angle = buffer[-1][5]
                    states["sit_up"], counters["sit_up"] = count_situp(
                        angle, states["sit_up"], counters["sit_up"]
                    )

        # Draw
        annotated = r.plot()

        label_map = {0: "Pull Up", 1: "Push Up", 2: "Sit Up"}
        label = label_map.get(prediction, "Detecting...")

        reps = (
            counters["push_up"] if prediction == 1
            else counters["pull_up"] if prediction == 0
            else counters["sit_up"] if prediction == 2
            else 0
        )

        cv2.putText(annotated, f"{label}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(annotated, f"Reps: {reps}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        out.write(annotated)

        progress.progress(frame_idx / total_frames)

    cap.release()
    out.release()

    st.success("Done!")

    st.video(output_path)