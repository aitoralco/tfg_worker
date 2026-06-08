import os
import re
from pathlib import Path

import cv2
import numpy as np


def generate_clean_whale_dataset(videos_dir, labels_dir, output_dir):
    videos_path = Path(videos_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    video_to_labels = {}

    for lf in label_files:
        match = re.search(r"^(.*)_(\d+)\.txt$", lf)
        if match:
            v_name, f_idx = match.groups()
            if v_name not in video_to_labels:
                video_to_labels[v_name] = []
            video_to_labels[v_name].append((int(f_idx), lf))

    for v_name, detections in video_to_labels.items():
        video_file = next(videos_path.glob(f"{v_name}.*"), None)
        if not video_file:
            continue

        print(f"Processing Clean Frames: {v_name}")
        cap = cv2.VideoCapture(str(video_file))
        detections.sort()  # Process frames in order for speed

        for f_idx, lf_name in detections:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            img_h, img_w = frame.shape[:2]

            with open(labels_path / lf_name, "r") as f:
                for obj_idx, line in enumerate(f.readlines()):
                    parts = list(map(float, line.split()))

                    a_num = 0

                    # If line has 9 (no conf) or 10 (with conf) values:
                    if len(parts) >= 9:
                        # Grab exactly the 8 coordinates (index 1 to 8)
                        coords = np.array(parts[1:9]).reshape((4, 2))
                        coords[:, 0] *= img_w
                        coords[:, 1] *= img_h
                        center = np.mean(coords, axis=0)

                        dist_01 = np.linalg.norm(coords[0] - coords[1])
                        dist_12 = np.linalg.norm(coords[1] - coords[2])

                        if dist_01 >= dist_12:
                            dy = coords[1][1] - coords[0][1]
                            dx = coords[1][0] - coords[0][0]

                        else:
                            dy = coords[2][1] - coords[1][1]
                            dx = coords[2][0] - coords[1][0]

                        angle = np.degrees(np.arctan2(dy, dx))

                        rotation_angle = 90 + angle

                        # Get dimensions for the square crop
                        width = np.linalg.norm(coords[0] - coords[1])
                        height = np.linalg.norm(coords[1] - coords[2])
                        side = int(max(width, height) * 1.3)  # 30% padding for whales
                    else:
                        continue

                    # Execute the Rotation and 1:1 Crop
                    # print(f"Angle: {angle}, in image: {v_name}, fame: {f_idx}")
                    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    rotated = cv2.warpAffine(frame, M, (img_w, img_h))

                    # GetRectSubPix handles sub-pixel centering beautifully
                    crop = cv2.getRectSubPix(rotated, (side, side), center)

                    if crop is not None:
                        save_dir = output_path / v_name
                        save_dir.mkdir(exist_ok=True)
                        save_name = (
                            f"{v_name}_f{f_idx}_obj{obj_idx}_a{rotation_angle}.jpg"
                        )
                        cv2.imwrite(str(save_dir / save_name), crop)

        cap.release()
    print("Clean Whale Dataset Ready!")
