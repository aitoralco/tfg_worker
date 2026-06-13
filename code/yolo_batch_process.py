#!/usr/bin/env python3
"""
Script para procesar imágenes con un modelo YOLO de clasificación en batch.
Guarda resultados detallados por imagen, imágenes anotadas y un resumen global.
"""

import os
import csv
import json
import argparse
from datetime import datetime
import cv2

from ultralytics import YOLO


def annotate_image(image, top1_name, top1_conf, top5_classes, top5_confs, class_names):
    """Anota la imagen con la clasificación predicha."""
    h, w = image.shape[:2]

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    label_main = f"{top1_name}  {top1_conf:.2%}"
    cv2.putText(image, label_main, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    top5_text = "  ".join(
        f"{class_names.get(c, c)}:{conf:.0%}" for c, conf in zip(top5_classes, top5_confs)
    )
    cv2.putText(image, top5_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return image


def process_images(model_path, images_dir, output_dir):
    """
    Procesa todas las imágenes en un directorio con un modelo YOLO de clasificación.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Directorio de imágenes no encontrado: {images_dir}")

    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, 'anotadas')
    os.makedirs(annotated_dir, exist_ok=True)

    print(f"[*] Cargando modelo: {model_path}")
    model = YOLO(model_path)
    class_names = model.names  # dict {0: 'nombre', 1: 'nombre', ...}
    print(f"[*] Clases del modelo: {class_names}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        print(f"[!] No se encontraron imágenes en: {images_dir}")
        return

    print(f"[*] Encontradas {len(image_files)} imágenes")

    total_images = len(image_files)
    classifications_by_class = {}
    all_confidences = []

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)
        print(f"[{idx}/{total_images}] Procesando: {image_file}")

        try:
            results = model.predict(source=image_path, verbose=False)
            result = results[0]

            probs = result.probs
            top1_class = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top5_classes = [int(c) for c in probs.top5]
            top5_confs = [float(c) for c in probs.top5conf]

            all_confidences.append(top1_conf)
            classifications_by_class[top1_class] = classifications_by_class.get(top1_class, 0) + 1

            image_data = {
                'archivo': image_file,
                'timestamp': datetime.now().isoformat(),
                'clase_predicha': top1_class,
                'clase_nombre': class_names.get(top1_class, f'clase_{top1_class}'),
                'confianza': round(top1_conf, 4),
                'top5': [
                    {
                        'clase': c,
                        'clase_nombre': class_names.get(c, f'clase_{c}'),
                        'confianza': round(conf, 4)
                    }
                    for c, conf in zip(top5_classes, top5_confs)
                ]
            }

            # JSON
            name = os.path.splitext(image_file)[0]
            with open(os.path.join(output_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(image_data, f, indent=2, ensure_ascii=False)

            # CSV (una fila por imagen con top-5)
            csv_path = os.path.join(output_dir, f"{name}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['clase_predicha', 'confianza',
                                 'top2_clase', 'top2_conf',
                                 'top3_clase', 'top3_conf',
                                 'top4_clase', 'top4_conf',
                                 'top5_clase', 'top5_conf'])
                row = [top1_class, round(top1_conf, 4)]
                for c, conf in zip(top5_classes[1:], top5_confs[1:]):
                    row += [c, round(conf, 4)]
                writer.writerow(row)

                # Imagen anotada
            img = cv2.imread(image_path)
            if img is not None:
                top1_name = class_names.get(top1_class, f'clase_{top1_class}')
                annotated = annotate_image(img.copy(), top1_name, top1_conf,
                                           top5_classes, top5_confs, class_names)
                cv2.imwrite(os.path.join(annotated_dir, image_file), annotated)

        except Exception as e:
            print(f"[!] Error procesando {image_file}: {e}")

    generate_summary(output_dir, total_images, classifications_by_class, all_confidences, class_names)

    print(f"\n[OK] Procesamiento completado")
    print(f"[OK] Resultados guardados en: {output_dir}")


def generate_summary(output_dir, total_images, classifications_by_class, all_confidences, class_names):
    """Genera el archivo RESUMEN.txt con estadísticas globales."""

    total_clasificadas = sum(classifications_by_class.values())
    summary_path = os.path.join(output_dir, 'RESUMEN.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RESUMEN DE CLASIFICACION YOLO\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ESTADISTICAS GENERALES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total de imagenes procesadas:   {total_images}\n")
        f.write(f"Total de imagenes clasificadas: {total_clasificadas}\n")
        f.write(f"Errores:                        {total_images - total_clasificadas}\n\n")

        if all_confidences:
            f.write("CONFIANZA DE LA PREDICCION (clase top-1)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Confianza promedio: {(sum(all_confidences)/len(all_confidences)*100):.2f}%\n")
            f.write(f"Confianza minima:   {(min(all_confidences)*100):.2f}%\n")
            f.write(f"Confianza maxima:   {(max(all_confidences)*100):.2f}%\n\n")

            f.write("DISTRIBUCION POR CLASE (top-1)\n")
            f.write("-" * 60 + "\n")
            for clase in sorted(classifications_by_class.keys()):
                count = classifications_by_class[clase]
                pct = count / total_clasificadas * 100
                bar = "#" * int(pct / 2)
                name = class_names.get(clase, f'clase_{clase}')
                f.write(f"{name} (id={clase}): {count:>5} imagenes  ({pct:5.2f}%)  {bar}\n")

        f.write("\nARCHIVOS GENERADOS\n")
        f.write("-" * 60 + "\n")
        f.write("- Imagenes anotadas:  carpeta 'anotadas/'\n")
        f.write("- Resultados por imagen: imagen.json / imagen.csv\n")
        f.write("- Este resumen:       RESUMEN.txt\n")
        f.write("\n" + "=" * 60 + "\n")

    print(f"[OK] Resumen guardado en: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Procesa imagenes con un modelo YOLO de clasificacion',
        epilog='Ejemplo: python yolo_batch_process.py -m best.pt -i imagenes/ -o resultados/'
    )
    parser.add_argument('-m', '--modelo', required=True, help='Ruta al archivo best.pt')
    parser.add_argument('-i', '--imagenes', required=True, help='Directorio con imagenes')
    parser.add_argument('-o', '--output', required=True, help='Directorio de salida')

    args = parser.parse_args()

    try:
        process_images(args.modelo, args.imagenes, args.output)
    except Exception as e:
        print(f"[!] Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
