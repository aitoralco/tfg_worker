def worker_test(text: str):
    # method to test the worker
    print("Worker got this text from api: {text}")

def process_video(video_id: int):
    from app.db.session import get_db
    from app.models.video_model import VideoModel
    from app.filesystem.filesystem_client import FileSystemClient
    from app.code.video_processor import VideoProcessor
    import os

    db_generator = get_db()
    db: Session = next(db_generator)

    # get video info from db
    video = db.query(VideoModel).get(video_id)
    
    # actualizar estatus a processing
    video.status_id = 4 # processing_dw
 
    #print(video.__dict__.copy())

    # obtener el filename, el user id y el group id
    filename = video.file_name
    user_id = video.user_id
    group_id = video.group_id

    # Crear directorio para el video y sus archivos 
    # (eliminando intentos anteriores de procesamiento de ser necesario)

    fs_client = FileSystemClient()

    # obtener el video
    download_path = f'./tmp/{filename}'
    
    fs_client.download_video_file(
        download_path=download_path,
        filename=filename,
        user_id=user_id,
        group_id=group_id
    )

    # Procesar video
    video_processor = VideoProcessor()

    result_path = f'/home/aitor/tfg_worker/app/tmp/results'

    os.makedirs(result_path, exist_ok=True)

    # DW detecting whales
    video_processor.whale_detecting(
        video_path=download_path,
        obb_video_result_path=result_path
    )
    

    # Cerrar transacción de DB para confirmar el estado de procesamiento
    db.commit()

    print(f"Successfuly got the video ID to process: {video_id}.")
    pass