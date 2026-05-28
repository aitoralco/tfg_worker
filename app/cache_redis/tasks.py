def worker_test(text: str):
    # method to test the worker
    print("Worker got this text from api: {text}")

def process_video(video_id: int):
    from app.db.session import engine
    from app.models.video_model import VideoModel

    # get video info from db
    video = db.query(VideoModel).get(video_id)
    
    # actualizar estatus a processing
    video.status_id = 3
    db.commit()

    # obtener el filename y el user
    filename = video.file_name
    user = video.user_id

    # obtener el video
    minio_client.download(filename)

    print(f"Successfuly got the video ID to process: {video_id}.")
    pass