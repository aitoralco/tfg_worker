"""
Class made to execute YOLO commands using Ultralytics library
for training, validating, testing, and predicting with YOLO models.

This class doesn't handle the lack of models or datasets; it assumes
that the paths provided are valid and accessible.

"""

from ultralytics import YOLO
import os
import logging
import time

custom_cache_path = './models/'

os.environ['ULTRALYTICS_CACHE_DIR'] = custom_cache_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class YoloExecutioner:
    def __init__(self, models_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("YoloExecutioner initialized with models path: %s", models_path)
        self.models_path = models_path


    def __load_model(self, model_source):
        # model_source can be a model name or a full path to .pt file
        # Load a YOLO model from the specified models path
        # good for testing, useless in practise
        # more of a private method

        if not model_source:
            raise ValueError("Model source must be provided.")

        # check if model_source is a path or just a name
        if os.path.isabs(model_source) or os.path.exists(model_source):
            self.logger.info("Model source is an absolute path or exists: %s", model_source)
            model_path = model_source
        else:
            self.logger.info("Model source is a relative name, constructing path: %s", model_source)
            model_path = os.path.join(self.models_path, model_source)

        self.logger.info("Loading model: %s", model_path)
        
        return YOLO(model_path)


    def train(self, yaml_path, epochs=100, batch=16, imgsz=640, model=None,
              workers=8, output_dir=None, model_name=None, cache=False,
              fliplr=0.5, flipud=0.5, degrees=5.0, nbs=64, patience=50, 
              lr0=0.01, lrf=0.01, optimizer='SGD'):

        self.logger.info("Starting training with model: %s", model)

        if not model:
            raise ValueError("Model must be provided for training.")
        model_object = self.__load_model(model)
        model_object.train(data=yaml_path, epochs=epochs, batch=batch, imgsz=imgsz,
                    workers=workers, project=output_dir, name=model_name, model=model,
                    cache=cache, patience=patience, lr0=lr0, lrf=lrf, optimizer=optimizer,
                    # --- AUGMENTATION PARAMS ---
                    fliplr=fliplr,
                    flipud=flipud,
                    degrees=degrees,
                    nbs=nbs
                    )


    def detect(self, trained_model_path, source, imgsz=640, conf=0.25, save_txt=False,
               save_conf=False, save_crop=False, save_frames=False, output_dir=None,
               save_result=False, stream=True, iou=0.45):
        """
        Perform detection using a trained YOLO model.
        trained_model_path: Path to the trained model weights .pt file.
        source: Source of the images/videos for detection, can be one or a list, or any format yolo accepts.
        imgsz: Image size for detection.
        conf: Confidence threshold for detections.
        save_txt: Whether to save detection results in a text file.
        save_conf: Whether to save confidence scores in the results.
        output_dir: Directory to save the detection results.
        save_result: Whether to save the resulting images/videos with detections.
        save_crop: Whether to save cropped detected objects.
        save_frames: Whether to save individual frames from videos.
        stream: streams one file at a time and saves one file results each time (lower RAM usage).
        """

        self.logger.info("Starting detection with model: %s", trained_model_path)

        model_object = self.__load_model(trained_model_path)

        self.logger.info("Performing prediction on source: %s", source)
        results = model_object.predict(source=source, imgsz=imgsz, conf=conf,
                             save_txt=save_txt, save_conf=save_conf,
                             project=output_dir, save_crop=save_crop,
                             save_frames=save_frames, save=save_result,
                             stream=stream, iou=iou
                             )
        
        for r in results if stream else []:
            pass

        self.logger.info("Detection completed.")