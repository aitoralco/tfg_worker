import boto3
from botocore.client import Config
from app.core.config import settings

class FileSystemClient:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            # --- MINIO Configuration ---
            endpoint_url=settings.FILESYSTEM_URL,
            aws_access_key_id=settings.FILESYSTEM_ID_KEY,
            aws_secret_access_key=settings.FILESYSTEM_ACCESS_KEY,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'} if settings.FORCE_PATH_STYLE == 'true' else {}
            ),
            region_name=settings.REGION_NAME
        )
        self.bucket = settings.BUCKET_NAME

    # descargar video y guardar en alguna ruta local
    def download_video_file(self, download_path: str, filename: str, user_id: int, group_id: int):
        
        object_name = f"{user_id}/{group_id}/{filename}"
        
        try:
            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=object_name,
                Filename=download_path
            )
            print(f"Successfully downloaded {object_name} to {download_path}")
            return True

        except Exception as e:
            print(f"Error downloading file from MinIO: {e}")
            raise

    # Subir video en stream a MinIO directamente
    def upload_video_stream(self, file_stream, filename: str, user_id: str, size: int, group_id: int):
        """Sube video en stream directamente al mino sin guardarlo en ningún directorio del backend"""

        object_name = f"{user_id}/{group_id}/{filename}"

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=object_name,
            Body=file_stream,
            ContentLength=size,
            ContentType="video/mp4"
        )

    # Subir video desde archivo a un directorio local
    def upload_video_file(self, file_path: str, object_name: str, user_id: str, processed: bool = False):
        """Upload a video file to the filesystem bucket."""
        try:
            # tries to create the bucket on upload
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
            except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
                pass

            if processed:
                proc = 'processed'
            else:
                proc = 'raw'

            object_key = f"{user_id}/{proc}/{object_name}"
            self.s3_client.upload_file(file_path, self.bucket, object_key)
            print(f"Uploaded {object_name} to bucket {self.bucket}")
        
        except Exception as e:
            print(f"Error uploading file: {e}")
            raise