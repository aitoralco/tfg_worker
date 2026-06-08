import logging
import os

from app.code.whale_cropper import generate_clean_whale_dataset
from app.code.yolo_executioner import YoloExecutioner
from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoProcessor:

    def whale_detecting(self, video_path: str, obb_video_result_path: str):
        """
        Fase DW: detecta cetáceos en el vídeo usando OBB YOLO.
        Genera:
          - Vídeo anotado con las OBB en obb_video_result_path/
          - Labels .txt por frame en obb_video_result_path/labels/
        """
        logger.info(f"Iniciando detección de ballenas en: {video_path}")

        model = YoloExecutioner("")
        model.detect(
            trained_model_path=settings.DW_MODEL_PATH,
            source=video_path,
            imgsz=1024,
            conf=0.5,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            output_dir=obb_video_result_path,
            save_result=True,
            stream=True,
            iou=0.25,
            vid_stride=1,
        )

        logger.info(f"Detección DW completada. Resultados en: {obb_video_result_path}")

    def whale_cropping(self, videos_dir: str, labels_dir: str, output_dir: str):
        """
        Fase EW (proceso intermedio): recorta y rota las ballenas detectadas.
        Toma las coordenadas OBB generadas por whale_detecting y extrae
        cada detección del vídeo original como imagen cuadrada alineada.

        Args:
            videos_dir: Directorio que contiene el vídeo original.
            labels_dir: Directorio con los .txt de labels generados por YOLO (dw_output/labels/).
            output_dir: Directorio donde se guardarán los recortes por vídeo.
        """
        logger.info(f"Iniciando recorte de ballenas. Labels: {labels_dir} → Output: {output_dir}")

        generate_clean_whale_dataset(
            videos_dir=videos_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
        )

        logger.info(f"Recorte EW completado. Recortes en: {output_dir}")

    def mark_detection(self, crops_dir: str, output_dir: str):
        """
        Fase DEM (modelo 2): clasifica o analiza los recortes individuales de ballenas.
        Placeholder — implementar cuando el modelo 2 esté listo.

        Args:
            crops_dir: Directorio con los recortes generados por whale_cropping.
            output_dir: Directorio donde se guardarán los resultados del modelo 2.
        """
        logger.info(f"Iniciando mark_detection (DEM). Crops: {crops_dir}")

        # TODO: implementar cuando el modelo DEM esté disponible
        # Estructura esperada:
        #
        model = YoloExecutioner("")
        model.detect(
            trained_model_path=settings.DEM_MODEL_PATH,
            source=crops_dir,
            output_dir=output_dir,
            save_txt=True,
            save_result=False,
            stream=True,
            save_crop=True,
            imgsz=1024,
            conf=0.5,
            save_conf=True,
            save_frames=False,
            iou=0.25,
            vid_stride=1
        )

        logger.info("mark_detection DEM: placeholder, sin modelo asignado aún.")