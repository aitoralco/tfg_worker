# Worker

[English](README.md) | **Español**

Worker en segundo plano del proyecto, construido con **RQ** (Redis Queue). Recoge los vídeos subidos a través del backend y los procesa con tres modelos **YOLO**: detecta los cetáceos, los recorta, detecta sus marcas y clasifica la especie. Los resultados se suben a **MinIO** y el progreso se guarda en **PostgreSQL**.

## Tecnología

- **Python 3.12**
- **RQ** + **Redis** (cola de trabajos compartida con el backend)
- **Ultralytics YOLO** + **PyTorch** (modelos), **OpenCV** (recortes)
- **ffmpeg** (convierte el vídeo anotado a `.mp4` H.264)
- **SQLAlchemy** (estado del vídeo en la BD), **boto3** (MinIO, compatible con S3)
- **pip** (dependencias en `requirements.txt`)

## Cómo funciona

El backend encola `process_video(video_id)` en Redis después de subir un vídeo. El worker recoge el trabajo y ejecuta estas fases, actualizando el estado del vídeo en la BD en cada una:

| Fase | Estado | Qué hace | Salida en MinIO |
|---|---|---|---|
| 0 | `processing_0` | Descarga el vídeo original de MinIO | — |
| DW | `processing_dw` | Detecta cetáceos con cajas orientadas (OBB) | `dw/` vídeo anotado + `dw/labels/` |
| EW | `processing_ew` | Recorta y alinea cada ballena detectada | `ew/` recortes |
| DEM | `processing_dem` | Detecta las marcas en los recortes | `dem/` |
| CW | `processing_dem` | Clasifica la especie de los recortes | `classification.json` |
| Fin | `processed` | Terminado | — |

Toda la salida se guarda en `{user_id}/{group_id}/{video_id}/` dentro del bucket. Si falla cualquier fase, el estado del vídeo pasa a `error`.

Cada trabajo usa su propia carpeta temporal, `/video_processing/job_{video_id}`. Los ficheros se borran en cuanto se suben, y la carpeta se elimina al final, aunque el trabajo falle.

## Estructura del proyecto

```
app/
├── main.py              # arranca el worker RQ (nombre "tfg-worker")
├── cache_redis/
│   ├── redis_engine.py  # conexión y cola de Redis
│   └── tasks.py         # trabajos: process_video, worker_test
├── code/
│   ├── video_processor.py   # un método por fase (DW, EW, DEM, CW)
│   ├── yolo_executioner.py  # wrapper de inferencia YOLO
│   └── whale_cropper.py     # recorte de OBB (EW)
├── core/config.py       # configuración
├── db/session.py        # engine, SessionLocal
├── filesystem/          # cliente de MinIO (descarga / subida)
├── models/              # modelos SQLAlchemy (copia de los del backend)
└── ai/trained_models/   # pesos YOLO entrenados (no están en el repo, ver Modelos)
requirements.txt
Dockerfile
```

## Modelos

> **Los pesos entrenados no están incluidos en este repositorio.** Los modelos se entrenaron con imágenes que no pertenecen al autor: se cedieron únicamente para este Trabajo de Fin de Grado (TFG). Ni las imágenes ni los modelos entrenados con ellas se pueden distribuir.

Para ejecutar el worker, coloca tus propios pesos en `app/ai/trained_models/` (la carpeta está ignorada por git):

| Variable | Fichero | Uso |
|---|---|---|
| `DW_MODEL_PATH` | `detect_whales/best.pt` | Detección de cetáceos (OBB) |
| `DEM_MODEL_PATH` | `detect_marks/best.pt` | Detección de marcas |
| `CW_MODEL_PATH` | `classify_whales/best.pt` | Clasificación de especies |

También puedes guardarlos en otro sitio y cambiar las variables `*_MODEL_PATH`. La imagen Docker copia la carpeta si existe en local.

Sin los pesos el worker arranca, pero todos los vídeos fallan en la fase DW y su estado pasa a `error`.

## Variables de entorno

La configuración la carga `app/core/config.py` desde variables de entorno o desde un fichero `.env` en la raíz del proyecto (ver `.env.example`).

| Grupo | Variables |
|---|---|
| Postgres | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_SERVER`, `POSTGRES_PORT`, `POSTGRES_DB` |
| Redis | `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` |
| MinIO | `FILESYSTEM_URL`, `FILESYSTEM_ID_KEY`, `FILESYSTEM_ACCESS_KEY`, `FORCE_PATH_STYLE`, `REGION_NAME`, `BUCKET_NAME` |
| Modelos | `DW_MODEL_PATH`, `DEM_MODEL_PATH`, `CW_MODEL_PATH` |
| Otras | `DEBUG` (por defecto `false`, imprime las consultas SQL) |

Todas son obligatorias (excepto `DEBUG`).

## Ejecución con Docker Compose (stack completo)

Desde `../tfg_dockercompose`:

```bash
docker compose up --build -d
```

El worker (`worker-rq`) solo arranca cuando `backend-migrate` ha terminado correctamente y Postgres, Redis y MinIO están healthy. Dentro del contenedor, las rutas de los modelos apuntan a `/worker/app/ai/trained_models/...` (ver `../tfg_dockercompose/.env.example`).

Comprobaciones útiles:

```bash
docker compose logs -f worker-rq    # trabajos y fases según se ejecutan
```

## Ejecución en local (desarrollo)

Hace falta Linux o WSL: RQ no funciona en Windows nativo.

1. Levanta los servicios de datos desde `../tfg_dockercompose`, con el esquema de la BD ya creado (ver `../tfg_backend/README.es.md`).
2. Crea `.env` a partir de `.env.example`, con las mismas credenciales que el `.env` del compose. Los hosts apuntan a `localhost` y las rutas de los modelos a los ficheros locales.
3. Comprueba que `ffmpeg` está instalado y que el usuario puede escribir en `/video_processing`.
4. Crea un entorno virtual e instala las dependencias:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Arranca el worker:
   ```bash
   python3 -m app.main
   ```

## Notas

- **Los nombres de los trabajos se comparten con el backend.** El backend encola `app.cache_redis.tasks.process_video` (tiene una función vacía con ese nombre en `tfg_backend/app/cache_redis/tasks.py`). Aquí la función tiene que existir con la misma ruta de módulo y la misma firma.
- **Los modelos de BD son una copia de los del backend.** Si el esquema cambia en el backend, hay que actualizar también `app/models/` aquí. El worker no ejecuta migraciones.
- **Los estados del vídeo se buscan por nombre** (`processing_dw`, `processed`…), así que tienen que existir en la BD. Los inserta el seed del backend (`tfg_backend/app/db/starter.py`).
- **Solo CPU.** El compose no reserva GPU, así que `requirements.txt` instala la versión CPU de PyTorch (`+cpu`, del índice de PyTorch) para que la imagen sea más pequeña. El procesamiento es lento. Para usar GPU, hay que instalar la versión CUDA de `torch`/`torchvision` y reservar la GPU en el compose.

## Licencia

© 2026 Aitor Alonso Coloma. Con licencia [MIT](LICENSE). La licencia cubre solo el código: los modelos entrenados y las imágenes usadas para entrenarlos no se incluyen y no están cubiertos por ella (ver [Modelos](#modelos)).
