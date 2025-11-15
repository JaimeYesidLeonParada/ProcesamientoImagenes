# -------------------- Imports y constantes --------------------
print("[STEP 01] Inicializando: cargando modulos y constantes")

from ollama import Client
import cv2
import numpy as np
import json

from utils import create_mask, find_largest_contour, get_warp_from_box, preprocess_plate

IMG_PATH = "placa.jpg"

# Umbrales
# H en OpenCV va 0..179; S,V en 0..255
HMIN, HMAX = 17, 27
SMIN, SMAX = 160, 255
VMIN, VMAX = 190, 255

# -------------------- Funciones de utilidad --------------------    

    
    
def call_ollama(img_path, model="moondream", prompt_user=None):
    print("[FNC call_ollama] Preparando cliente ollama para:", img_path)
    
    if prompt_user is None:
        prompt_user = ("Read the plate and the city name from the image. "
                       "Return both separated by comma. An example of the result is: 'XYZ 123 , PASTO DC'")

    try:
        client = Client()
        print("[FNC call_ollama] Cliente creado. Enviando solicitud...")
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are an OCR that reads car plates and city text below."},
                {"role": "user", "content": prompt_user, "images": [img_path]}
            ],
            options={"temperature": 0.0, "num_predict": 32}
        )
    except Exception as e:
        print("[FNC call_ollama] ERROR: fallo comunicacion con ollama:", str(e))
        return None

    try:
        content = resp.get("message", {}).get("content", None)
        if content:
            print("[FNC call_ollama] Respuesta recibida")
            return content
        else:
            print("[FNC call_ollama] ERROR: respuesta sin contenido util. Respuesta cruda:")
            print(json.dumps(resp, indent=2))
            return None
    except Exception as e:
        print("[FNC call_ollama] ERROR procesando la respuesta:", str(e))
        print("Respuesta cruda:", resp)
        return None

# -------------------- Main y ejecucion --------------------
def main():
    print("[MAIN] Inicio del flujo principal")

    # STEP 02: cargar imagen
    print("[MAIN] STEP 02 - Cargando imagen:", IMG_PATH)
    bgr = cv2.imread(IMG_PATH)
    if bgr is None:
        print("[MAIN] [ERROR] No se pudo leer la imagen:", IMG_PATH)
        raise SystemExit("No se pudo leer la imagen.")
    h_img, w_img = bgr.shape[:2]
    print(f"[MAIN] Imagen cargada correctamente ({w_img}x{h_img})")

    # STEP 03: crear mascara
    print("[MAIN] STEP 03 - Generando mascara")
    mask, kernel = create_mask(bgr, HMIN, HMAX, SMIN, SMAX, VMIN, VMAX)

    # STEP 04: overlay/blend
    print("[MAIN] STEP 04 - Creando overlay y blend")
    overlay = bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    blend = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
    cv2.imwrite("debug_overlay.jpg", overlay)
    cv2.imwrite("debug_blend.jpg", blend)
    print("[MAIN] Debug overlay/blend guardados")

    # STEP 05: contornos
    print("[MAIN] STEP 05 - Buscando contornos")
    cnt, box = find_largest_contour(mask, kernel)
    if cnt is None:
        print("[MAIN] [ERROR] No se encontraron contornos")
        raise SystemExit("No se encontraron contornos.")
    area = int(cv2.contourArea(cnt))
    print(f"[MAIN] Contorno seleccionado. Area = {area} pixeles")
    debug_cnt = bgr.copy()
    cv2.drawContours(debug_cnt, [box.astype(int)], 0, (0,255,0), 2)
    cv2.imwrite("debug_contour_box.jpg", debug_cnt)
    print("[MAIN] Debug contour box guardado")

    # STEP 06: warp
    print("[MAIN] STEP 06 - Generando warp")
    warp, M, size = get_warp_from_box(box, bgr, target_h=240, min_w=120)
    if warp is None:
        print("[MAIN] [ERROR] No se pudo generar warp")
        raise SystemExit("No se pudo crear warp")
    target_w, target_h = size
    cv2.imwrite("plate_warp.jpg", warp)
    print(f"[MAIN] Imagen warp guardada: plate_warp.jpg ({target_w}x{target_h})")

    # STEP 07: preprocesado
    print("[MAIN] STEP 07 - Preprocesando placa")
    out_path = preprocess_plate(warp, out_path="plate_prepoc.jpg")
    if out_path is None:
        print("[MAIN] [ERROR] Preprocesado fallido")
        raise SystemExit("Preprocesado fallido")
    print("[MAIN] Imagen preprocesada disponible en", out_path)

    # STEP 08: OCR
    print("[MAIN] STEP 08 - Llamando OCR")
    ocr_result = call_ollama(out_path)
    if ocr_result is None:
        print("[MAIN] [ERROR] OCR no devolvio resultado util")
        raise SystemExit("Error en llamada al servicio OCR")
    print("[MAIN] Resultado OCR:")
    print(ocr_result)

    print("[MAIN] Flujo completado correctamente")

if __name__ == "__main__":
    main()
