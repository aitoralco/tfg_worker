# Worker

**English** | [Español](README.es.md)

Background worker of the project, built with **RQ** (Redis Queue). It takes the videos uploaded through the backend and processes them with three **YOLO** models: it detects cetaceans, crops them, detects their marks and classifies the species. Results are uploaded to **MinIO** and the progress is written to **PostgreSQL**.

## Technology

- **Python 3.12**
- **RQ** + **Redis** (job queue shared with the backend)
- **Ultralytics YOLO** + **PyTorch** (models), **OpenCV** (cropping)
- **ffmpeg** (converts the annotated video to H.264 `.mp4`)
- **SQLAlchemy** (video status in the DB), **boto3** (MinIO, S3 compatible)
- **pip** (dependencies in `requirements.txt`)

## How it works

The backend enqueues `process_video(video_id)` in Redis after a video is uploaded. The worker picks up the job and runs these phases, updating the video status in the DB at each one:

| Phase | Status | What it does | Output in MinIO |
|---|---|---|---|
| 0 | `processing_0` | Downloads the original video from MinIO | — |
| DW | `processing_dw` | Detects cetaceans with oriented boxes (OBB) | `dw/` annotated video + `dw/labels/` |
| EW | `processing_ew` | Crops and aligns each detected whale | `ew/` crops |
| DEM | `processing_dem` | Detects marks on the crops | `dem/` |
| CW | `processing_dem` | Classifies the species of the crops | `classification.json` |
| End | `processed` | Done | — |

All the output is stored under `{user_id}/{group_id}/{video_id}/` in the bucket. If any phase fails, the video status is set to `error`.

Each job works in its own temporary folder, `/video_processing/job_{video_id}`. Files are deleted as soon as they are uploaded, and the folder is removed at the end, even if the job fails.

## Project structure

```
app/
├── main.py              # starts the RQ worker (name "tfg-worker")
├── cache_redis/
│   ├── redis_engine.py  # Redis connection and queue
│   └── tasks.py         # jobs: process_video, worker_test
├── code/
│   ├── video_processor.py   # one method per phase (DW, EW, DEM, CW)
│   ├── yolo_executioner.py  # YOLO inference wrapper
│   └── whale_cropper.py     # OBB cropping (EW)
├── core/config.py       # settings
├── db/session.py        # engine, SessionLocal
├── filesystem/          # MinIO client (download / upload)
├── models/              # SQLAlchemy models (copy of the backend ones)
└── ai/trained_models/   # trained YOLO weights (not in the repo, see Models)
requirements.txt
Dockerfile
```

## Models

> **The trained weights are not included in this repository.** The models were trained with images that do not belong to the author: they were provided only for this final degree project (TFG). Neither the images nor the models trained with them can be distributed.

To run the worker, place your own weights in `app/ai/trained_models/` (the folder is ignored by git):

| Variable | File | Use |
|---|---|---|
| `DW_MODEL_PATH` | `detect_whales/best.pt` | Cetacean detection (OBB) |
| `DEM_MODEL_PATH` | `detect_marks/best.pt` | Mark detection |
| `CW_MODEL_PATH` | `classify_whales/best.pt` | Species classification |

You can also store them somewhere else and change the `*_MODEL_PATH` variables. The Docker image copies the folder if it exists locally.

Without the weights the worker starts, but every video fails in the DW phase and its status is set to `error`.

## Environment variables

Settings are loaded by `app/core/config.py` from environment variables or from a `.env` file in the project root.

| Group | Variables |
|---|---|
| Postgres | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_SERVER`, `POSTGRES_PORT`, `POSTGRES_DB` |
| Redis | `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` |
| MinIO | `FILESYSTEM_URL`, `FILESYSTEM_ID_KEY`, `FILESYSTEM_ACCESS_KEY`, `FORCE_PATH_STYLE`, `REGION_NAME`, `BUCKET_NAME` |
| Models | `DW_MODEL_PATH`, `DEM_MODEL_PATH`, `CW_MODEL_PATH` |
| Other | `DEBUG` (default `false`, prints SQL queries) |

All of them are required (except `DEBUG`).

## Running with Docker Compose (full stack)

From `../tfg_dockercompose`:

```bash
docker compose up --build -d
```

The worker (`worker-rq`) only starts after `backend-migrate` finishes successfully and Postgres, Redis and MinIO are healthy. In the container, the model paths point to `/worker/app/ai/trained_models/...` (see `../tfg_dockercompose/.env.example`).

Useful checks:

```bash
docker compose logs -f worker-rq    # jobs and phases as they run
```

## Running locally (development)

Linux or WSL is required: RQ does not run on native Windows.

1. Start the data services from `../tfg_dockercompose`, with the DB schema already created (see `../tfg_backend/README.md`).
2. Create a `.env` with the variables above, pointing hosts to `localhost` and the model paths to the local files, e.g. `DW_MODEL_PATH=app/ai/trained_models/detect_whales/best.pt`.
3. Make sure `ffmpeg` is installed and that the user can write to `/video_processing`.
4. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Start the worker:
   ```bash
   python3 -m app.main
   ```

## Notes

- **Job names are shared with the backend.** The backend enqueues `app.cache_redis.tasks.process_video` (it has an empty function with that name in `tfg_backend/app/cache_redis/tasks.py`). The function must exist with the same module path and signature here.
- **The DB models are a copy of the backend ones.** If the schema changes in the backend, update `app/models/` here too. The worker does not run migrations.
- **Video status IDs are fixed in `tasks.py`** (`1` = `error` … `7` = `processed`). They rely on the insertion order of the backend seed (`tfg_backend/app/db/starter.py`).
- **No GPU in Docker Compose.** The compose file does not reserve a GPU, so inference runs on the CPU and processing is slow.
