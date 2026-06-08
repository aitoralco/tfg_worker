import os

import torch
import whale_cropper
from dotenv import load_dotenv
from ultralytics import YOLO
from video_adjuster import VideoAdjuster
from yolo_executioner import YoloExecutioner

load_dotenv()

MODELS_PATH = os.getenv("MODELS_PATH")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# model = YOLO(MODELS_PATH + 'yolo11n.pt')

# model_ = YoloExecutioner(MODELS_PATH).load_model('yolo11n.pt')

if __name__ == "__main__":
    model_ = YoloExecutioner(MODELS_PATH)
    yaml_dir_path = os.getenv("DATASET_YAML_PATH")
    yaml_path = os.path.join(yaml_dir_path, "dataset_05.yaml")
    output_dir = os.getenv("OUTPUTS_PATH")
    results_path = os.getenv("RESULTS_PATH")
    source_path = "E:/00_tfg/videos/testing-videos"
    data_path = "E:/00_tfg/code/tfg_utils/datasets/dataset_whale_classify_01"

    train = False
    detect = True
    adjust_videos = False
    crop_whales = True
    detect_marks = True

    if train:
        model_.train(
            # yaml_path=yaml_path, # for no cls models
            data=data_path,
            epochs=1000,
            batch=2,
            imgsz=520,
            model="yolo26n-cls.pt",
            output_dir=output_dir,
            workers=4,
            model_name="yolo26n_cls520p_1000e_augmented_nbs64_adamw",
            cache="disk",
            fliplr=0.5,
            flipud=0.5,
            degrees=5.0,
            nbs=64,
            patience=50,
            lr0=0.0005,  # default 0.01
            lrf=0.01,  # default 0.01
            optimizer="AdamW",  # default 'SGD'
        )

    if adjust_videos:
        videos_source = "E:/00_tfg/videos/testing-videos"
        videos_destination = "E:/00_tfg/videos/adjusted-testing-videos"
        video_machine = VideoAdjuster(videos_source, videos_destination)
        video_machine.adjust_videos()

    # last model used: 11sobb_d06_1024p_1000e_adamw_v22

    if detect:
        model_.detect(
            trained_model_path=os.path.join(
                output_dir, "yolo26s_obb_d07_1024p_1000e_musgd_v1", "weights", "best.pt"
            ),
            source=source_path,
            imgsz=1024,
            conf=0.5,
            save_txt=True,
            save_conf=True,
            save_crop=False,  # False for OBB
            save_frames=False,  # True for OBB
            output_dir=results_path,
            save_result=False,
            stream=True,
            iou=0.25,
            vid_stride=1,
        )

    if crop_whales:
        predict = 10
        labels_path = f"./runs/obb/results/predict{predict}/labels"
        mark_dataset_output_path = f"./croped/testing_videos_01"
        whale_cropper.generate_clean_whale_dataset(
            source_path, labels_path, mark_dataset_output_path
        )

    if detect_marks:
        cropped_img_source_path = f"./croped/testing_videos_01/DJI_0573"
        model_.detect(
            trained_model_path=os.path.join(
                output_dir, "26s_d01m_1024p_1000e_musgd_v1", "weights", "best.pt"
            ),
            source=cropped_img_source_path,
            imgsz=1024,
            conf=0.5,
            save_txt=True,
            save_conf=True,
            save_crop=True,  # False for OBB
            save_frames=False,  # True for OBB
            output_dir=results_path,
            save_result=False,
            stream=True,
            iou=0.25,
            vid_stride=1,
        )
