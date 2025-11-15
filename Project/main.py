# -------------------- Imports y constantes --------------------
print("[STEP 01] Inicializando: cargando modulos y constantes")

import cv2
import os
import numpy as np
from ocr import call_ollama
from utils import create_mask, find_largest_contour, get_warp_from_box, preprocess_plate
from ocr_clean import clean_ocr_text


IMG_PATH = "placa.jpg"

# Umbrales
# H en OpenCV va 0..179; S,V en 0..255
HMIN, HMAX = 17, 27
SMIN, SMAX = 160, 255
VMIN, VMAX = 190, 255

def resize_image(img, target_w=350):
    h, w = img.shape[:2]
    scale = target_w / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(img, (350, 250), interpolation=cv2.INTER_CUBIC)
    return resized

# -------------------- Main y ejecucion --------------------
def main(folder="data/placas"):
    print("[MAIN] Inicio del flujo en carpeta:", folder)

    # listar archivos de imagen en la carpeta
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        print("[MAIN] No se encontraron imagenes en", folder)
        return

    for fname in files:
        path = os.path.join(folder, fname)
        print("\n[MAIN] Procesando:", path)

        bgr = cv2.imread(path)
                
        if bgr is None:
            print("[MAIN] [ERROR] No se pudo leer la imagen:", path)
            continue
        bgr = resize_image(bgr, target_w = 500)

        # STEP 03: mascara
        mask, kernel = create_mask(bgr, HMIN, HMAX, SMIN, SMAX, VMIN, VMAX)

        # STEP 05: contornos
        cnt, box = find_largest_contour(mask, kernel)
        if cnt is None:
            print("[MAIN] [ERROR] No se encontraron contornos en", path)
            continue

        # STEP 06: warp
        warp, M, size = get_warp_from_box(box, bgr, target_h=240, min_w=120)
        if warp is None:
            print("[MAIN] [ERROR] No se pudo generar warp en", path)
            continue

        # STEP 07: preprocesado
        out_path = preprocess_plate(warp, out_path=f"{fname}_prep.jpg")
        if out_path is None:
            print("[MAIN] [ERROR] Preprocesado fallido en", path)
            continue

        # STEP 08: OCR
        ocr_result = call_ollama(out_path)
        if ocr_result is None:
            print("[MAIN] [ERROR] OCR fallo en", path)
            continue

        # limpieza postprocesado
        plate_fixed, city = clean_ocr_text(ocr_result)
        print("[MAIN] Resultado OLLAMA:", ocr_result)
        print("[MAIN] Resultado OCR limpio:", plate_fixed, ",", city)

    print("\n[MAIN] Flujo completado para carpeta:", folder)

if __name__ == "__main__":
    main("data/placas")  # ajusta la ruta a tu carpeta real
