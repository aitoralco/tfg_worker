import logging
import os
import shutil

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def worker_test(text: str):
    print(f"Worker got this text from api: {text}")


def process_video(video_id: int):
    from app.db.session import get_db
    from app.models.video_model import VideoModel
    from app.filesystem.filesystem_client import FileSystemClient
    from app.code.video_processor import VideoProcessor

    db_generator = get_db()
    db: Session = next(db_generator)

    # Directorio temporal del job — aislado por video_id para evitar colisiones
    job_tmp_dir = f"/tmp/job_{video_id}"

    try:
        # --- Obtener info del video ---
        video = db.query(VideoModel).get(video_id)
        if video is None:
            logger.error(f"Video {video_id} not found in DB.")
            return

        filename = video.file_name
        user_id = video.user_id
        group_id = video.group_id

        fs_client = FileSystemClient()
        video_processor = VideoProcessor()

        # --- Crear directorio temporal del job ---
        os.makedirs(job_tmp_dir, exist_ok=True)

        # -------------------------------------------------------
        # FASE 0: Descarga del video
        # -------------------------------------------------------
        _set_status(db, video, status_id=3)  # processing_0

        video_path = os.path.join(job_tmp_dir, filename)
        fs_client.download_video_file(
            download_path=video_path,
            filename=filename,
            user_id=user_id,
            group_id=group_id,
        )
        logger.info(f"[{video_id}] Video descargado en {video_path}")

        # -------------------------------------------------------
        # FASE 1: Detección de ballenas (modelo DW)
        # -------------------------------------------------------
        _set_status(db, video, status_id=4)  # processing_dw

        dw_output_dir = os.path.join(job_tmp_dir, "dw_output")
        os.makedirs(dw_output_dir, exist_ok=True)

        video_processor.whale_detecting(
            video_path=video_path,
            obb_video_result_path=dw_output_dir,
        )

        # YOLO guarda: vídeo anotado en dw_output_dir/ y labels en dw_output_dir/labels/
        dw_labels_dir = os.path.join(dw_output_dir, "predict", "labels")
        dw_annotated_video = _find_annotated_video(dw_output_dir, filename)
        dw_annotated_video = _convert_to_mp4(dw_annotated_video)

        # Subir vídeo anotado y labels a MinIO
        fs_client.upload_file(
            local_path=dw_annotated_video,
            remote_key=f"{user_id}/{group_id}/{video_id}/dw/{os.path.basename(dw_annotated_video)}",
        )
        fs_client.upload_directory(
            local_dir=dw_labels_dir,
            remote_prefix=f"{user_id}/{group_id}/{video_id}/dw/labels/",
        )
        logger.info(f"[{video_id}] Fase DW completada y subida a MinIO.")

        # -------------------------------------------------------
        # FASE 2: Proceso intermedio — recorte de ballenas (EW)
        # -------------------------------------------------------
        _set_status(db, video, status_id=5)  # processing_ew

        ew_output_dir = os.path.join(job_tmp_dir, "ew_output")
        os.makedirs(ew_output_dir, exist_ok=True)

        video_processor.whale_cropping(
            videos_dir=os.path.dirname(video_path),
            labels_dir=dw_labels_dir,
            output_dir=ew_output_dir,
        )

        ew_crops_subdir = os.path.join(ew_output_dir, os.path.splitext(filename)[0])

        # Subir recortes a MinIO
        fs_client.upload_directory(
            local_dir=ew_output_dir,
            remote_prefix=f"{user_id}/{group_id}/{video_id}/ew/",
        )
        logger.info(f"[{video_id}] Fase EW completada y subida a MinIO.")

        # -------------------------------------------------------
        # FASE 3: Modelo 2 (DEM) — placeholder
        # -------------------------------------------------------
        _set_status(db, video, status_id=6)  # processing_dem

        dem_output_dir = os.path.join(job_tmp_dir, "dem_output")
        os.makedirs(dem_output_dir, exist_ok=True)

        video_processor.mark_detection(
            crops_dir=ew_crops_subdir,
            output_dir=dem_output_dir,
        )

        fs_client.upload_directory(
            local_dir=dem_output_dir,
            remote_prefix=f"{user_id}/{group_id}/{video_id}/dem/",
        )
        logger.info(f"[{video_id}] Fase DEM completada y subida a MinIO.")

        # -------------------------------------------------------
        # FIN: Marcar como procesado y limpiar temporales
        # -------------------------------------------------------
        _set_status(db, video, status_id=7)  # processed
        logger.info(f"[{video_id}] Procesado con éxito.")

    except Exception as e:
        logger.exception(f"[{video_id}] Error durante el procesamiento: {e}")
        try:
            video = db.query(VideoModel).get(video_id)
            if video:
                _set_status(db, video, status_id=1)  # error
        except Exception:
            pass

    finally:
        # Limpiar temporales siempre, tanto en éxito como en error
        if os.path.exists(job_tmp_dir):
            # shutil.rmtree(job_tmp_dir)
            logger.info(f"[{video_id}] Temporales eliminados: {job_tmp_dir}")
        try:
            db.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_status(db: Session, video, status_id: int):
    """Actualiza el status del video y hace commit inmediato."""
    video.status_id = status_id
    db.commit()


def _find_annotated_video(output_dir: str, original_filename: str) -> str:
    """
    Busca el vídeo anotado que YOLO genera en output_dir.
    YOLO mantiene el nombre original del vídeo fuente.
    """
    predict_dir = os.path.join(output_dir, "predict")
    base_name = os.path.splitext(original_filename)[0]

    search_dir = predict_dir if os.path.exists(predict_dir) else output_dir

    for f in os.listdir(search_dir):
        if f.startswith(base_name) and f.lower().endswith((".mp4", ".avi", ".mov")):
            return os.path.join(search_dir, f)
    raise FileNotFoundError(
        f"No se encontró vídeo anotado para '{original_filename}' en '{output_dir}'"
    )


def _convert_to_mp4(input_path: str) -> str:
    """
    Convierte el vídeo de salida de YOLO a .mp4 (H.264) y añade sufijo _annotated.
    Elimina el fichero original tras la conversión.
    """
    import subprocess

    base = os.path.splitext(input_path)[0]
    output_path = f"{base}_annotated.mp4"

    logger.info(f"Convirtiendo vídeo a mp4: {input_path} → {output_path}")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    logger.debug(f"ffmpeg output: {result.stderr}")
    os.remove(input_path)
    logger.info(f"Conversión completada: {output_path}")
    return output_path