import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_clean_whale_dataset(videos_dir, labels_dir, output_dir):
    videos_path = Path(videos_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    video_to_labels = {}

    logger.info(f"EW: {len(label_files)} label files encontrados en {labels_dir}")

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

        # Índice: frame_idx -> lista de label files — permite lectura secuencial sin seeks
        detections.sort()
        detection_map = {}
        for f_idx, lf_name in detections:
            detection_map.setdefault(f_idx, []).append(lf_name)

        max_frame = detections[-1][0]
        total_detections = len(detections)
        logger.info(f"EW [{v_name}]: {total_detections} frames con detecciones, leyendo hasta frame {max_frame}")

        cap = cv2.VideoCapture(str(video_file))
        current_frame = 0
        crops_saved = 0

        while current_frame <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in detection_map:
                img_h, img_w = frame.shape[:2]

                for lf_name in detection_map[current_frame]:
                    with open(labels_path / lf_name, "r") as f:
                        for obj_idx, line in enumerate(f.readlines()):
                            parts = list(map(float, line.split()))

                            if len(parts) >= 9:
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

                                width = np.linalg.norm(coords[0] - coords[1])
                                height = np.linalg.norm(coords[1] - coords[2])
                                side = int(max(width, height) * 1.3)
                            else:
                                continue

                            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                            rotated = cv2.warpAffine(frame, M, (img_w, img_h))
                            crop = cv2.getRectSubPix(rotated, (side, side), center)

                            if crop is not None:
                                save_dir = output_path / v_name
                                save_dir.mkdir(exist_ok=True)
                                save_name = f"{v_name}_f{current_frame}_obj{obj_idx}_a{rotation_angle}.jpg"
                                cv2.imwrite(str(save_dir / save_name), crop)
                                crops_saved += 1

            current_frame += 1

        cap.release()
        logger.info(f"EW [{v_name}]: completado — {crops_saved} recortes guardados")

    logger.info(f"EW: proceso finalizado para {len(video_to_labels)} vídeo(s)")
