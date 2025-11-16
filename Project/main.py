# -------------------- Imports y constantes --------------------
import cv2
import os
import numpy as np
import time   # para medir tiempos
import csv    # para guardar resultados en CSV
from ocr import call_ollama
from utils import create_mask, find_largest_contour, get_warp_from_box, preprocess_plate
from ocr_clean import clean_ocr_text

# Umbrales
HMIN, HMAX = 17, 27
SMIN, SMAX = 160, 255
VMIN, VMAX = 190, 255

# -------------------- Main y ejecucion --------------------
def main(folder="data/placas", csv_out="resultados.csv"):
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        print("[MAIN] No se encontraron imagenes en", folder)
        return

    # preparar archivo CSV (cabecera si no existe)
    write_header = not os.path.exists(csv_out)
    with open(csv_out, mode="a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(["Archivo", "OCR_bruto", "OCR_limpio", "Ciudad", "Tiempo_s"])

        for fname in files:
            path = os.path.join(folder, fname)
            print("\n[MAIN] Procesando:", path)

            bgr = cv2.imread(path)
            if bgr is None:
                print("[MAIN] [ERROR] No se pudo leer la imagen:", path)
                continue

            mask, kernel = create_mask(bgr, HMIN, HMAX, SMIN, SMAX, VMIN, VMAX)
            cnt, box = find_largest_contour(mask, kernel)
            if cnt is None:
                print("[MAIN] [ERROR] No se encontraron contornos en", path)
                continue

            warp, M, size = get_warp_from_box(box, bgr, target_h=240, min_w=120)
            if warp is None:
                print("[MAIN] [ERROR] No se pudo generar warp en", path)
                continue
            
            name, ext = os.path.splitext(fname)
            
            out_path = preprocess_plate(warp, out_path=f"data/placasprepro/{name}_prep.jpg")
            if out_path is None:
                print("[MAIN] [ERROR] Preprocesado fallido en", path)
                continue

            # STEP 08: OCR con medicion de tiempo
            start_time = time.time()
            ocr_result = call_ollama(out_path)
            end_time = time.time()
            elapsed = end_time - start_time

            if ocr_result is None:
                print("[MAIN] [ERROR] OCR fallo en", path)
                continue

            plate_fixed, city = clean_ocr_text(ocr_result)

            # salida organizada para informe
            print(f"[RESULT] Archivo: {fname} | OCR bruto: {ocr_result} | OCR limpio: {plate_fixed}, {city} | Tiempo: {elapsed:.3f} s")

            # guardar en CSV
            writer.writerow([fname, ocr_result, plate_fixed, city, f"{elapsed:.3f}"])

    print("\n[MAIN] Flujo completado para carpeta:", folder)

if __name__ == "__main__":
    main("data/placas")
