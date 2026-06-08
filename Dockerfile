# Dockerfile for worker
# Imagen de python
FROM python:3.12-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /worker

# Copiar archivo de requirements.txt
COPY ./requirements.txt /worker/requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade -r /worker/requirements.txt

# Copiar el resto de la aplicacion
COPY . /worker

# Exponer el puerto 8500
EXPOSE 8500

# Comando para arrancar el worker
CMD ["python","-m" "app.main"]