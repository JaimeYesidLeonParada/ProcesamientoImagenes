# -------------------- Imports y constantes --------------------
print("[STEP 01] Inicializando: cargando modulos y constantes")

import cv2
import numpy as np

IMG_PATH = "placa.jpg"

# Umbrales
# H en OpenCV va 0..179; S,V en 0..255
HMIN, HMAX = 17, 27
SMIN, SMAX = 160, 255
VMIN, VMAX = 190, 255

# -------------------- Funciones de utilidad --------------------
print("[STEP FNC] Definiendo funciones de utilidad")

def create_mask(bgr, hmin, hmax, smin, smax, vmin, vmax):
    # convertir a HSV y crear mascara segun umbrales
    print("[FNC create_mask] Convirtiendo BGR a HSV")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    h, w, _ = hsv.shape
    k = max(3, min(15, w // 100))  # escala: 3..15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    print(f"[FNC create_mask] Kernel: {k}x{k}")

    mask = cv2.inRange(hsv, lower, upper)
    white_pixels = int(mask.sum() / 255)
    print(f"[FNC create_mask] Mascara inicial. Pixeles blancos: {white_pixels} / {w*h}")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    print("[FNC create_mask] Operaciones morfologicas aplicadas: OPEN, CLOSE")

    return mask, kernel

def find_largest_contour(mask, kernel):
    # cerrar small gaps antes de buscar contornos
    print("[FNC find_largest_contour] Aplicando CLOSE adicional y buscando contornos")
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[FNC find_largest_contour] No se encontraron contornos")
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    print(f"[FNC find_largest_contour] Contorno mayor encontrado. Area = {int(area)}")

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    print("[FNC find_largest_contour] Caja minima calculada")

    return cnt, box
    
def get_warp_from_box(box, image, target_h=240, min_w=120):
    # ordenar puntos como tl, tr, br, bl
    print("[FNC get_warp_from_box] Ordenando puntos fuente")
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    print("[FNC get_warp_from_box] Puntos fuente ordenados:", src.tolist())

    # calcular ancho y alto del rectificado
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW = max(wA, wB)
    maxH = max(hA, hB)
    aspect = maxW / maxH if maxH > 0 else 4.0
    target_w = int(max(min_w, target_h * aspect))
    print(f"[FNC get_warp_from_box] Tamano destino: {target_w}x{target_h} (aspecto {aspect:.2f})")

    # puntos destino y homografia
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    print("[FNC get_warp_from_box] Matriz de transformacion calculada")

    # aplicar warp
    warp = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    if warp is None or warp.size == 0:
        print("[FNC get_warp_from_box] Error: warp vacio")
        return None, None, None
    print("[FNC get_warp_from_box] Warp generado correctamente")
    return warp, M, (target_w, target_h)

def preprocess_plate(img, out_path="plate_prepoc.jpg", target_h=None):
    # img: imagen BGR rectificada (warp)
    # out_path: ruta donde se guardara la imagen resultante
    print("[FNC preprocess_plate] Iniciando preprocesado")

    if img is None:
        print("[FNC preprocess_plate] Error: imagen de entrada es None")
        return None

    print("[FNC preprocess_plate] Conversion a gris")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("[FNC preprocess_plate] Aplicando CLAHE")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    print("[FNC preprocess_plate] Aplicando filtro bilateral (reduccion de ruido)")
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    print("[FNC preprocess_plate] Aplicando desenfoque y realce (sharpen)")
    blur  = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    print(f"[FNC preprocess_plate] Convirtiendo a BGR y guardando en: {out_path}")
    out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    ok = cv2.imwrite(out_path, out)
    if not ok:
        print("[FNC preprocess_plate] Error guardando archivo:", out_path)
        return None

    print("[FNC preprocess_plate] Preprocesado completado:", out_path)
    return out_path


# -------------------- Cargar imagen --------------------
print("[STEP 02] Cargando imagen:", IMG_PATH)

bgr = cv2.imread(IMG_PATH)
if bgr is None:
    print("[ERROR] No se pudo leer la imagen:", IMG_PATH)
    raise SystemExit("No se pudo leer la imagen.")
else:
    h_img, w_img = bgr.shape[:2]
    print(f"[STEP 02] Imagen cargada correctamente ({w_img}x{h_img})")


# -------------------- Convertir a HSV y crear mascara --------------------
print("[STEP 03] Generando mascara usando create_mask")
mask, kernel = create_mask(bgr, HMIN, HMAX, SMIN, SMAX, VMIN, VMAX)

# -------------------- Overlay y blend --------------------
print("[STEP 04] Creando overlay y blend para debug")

# crear overlay marcando pixeles detectados en rojo
overlay = bgr.copy()
overlay[mask > 0] = (0, 0, 255)

# mezclar original y overlay para visualizacion

blend = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
print("[STEP 04] Overlay y blend creados")


# -------------------- Deteccion de contornos y seleccion --------------------
print("[STEP 05] Buscando contornos usando find_largest_contour")
cnt, box = find_largest_contour(mask, kernel)
if cnt is None:
    print("[ERROR] No se encontraron contornos en la mascara")
    raise SystemExit("No se encontraron contornos.")
else:
    area = int(cv2.contourArea(cnt))
    print(f"[STEP 05] Contorno seleccionado. Area = {area} pixeles")
    print("[STEP 05] Caja minima (RAW):", box.tolist())
   
# -------------------- Ordenar puntos y calcular homografia --------------------
print("[STEP 06] Generando warp usando get_warp_from_box")
warp, M, size = get_warp_from_box(box, bgr, target_h=240, min_w=120)
if warp is None:
    print("[ERROR] No se pudo generar warp a partir de la caja")
    raise SystemExit("No se pudo crear warp")
else:
    target_w, target_h = size
    cv2.imwrite("plate_warp.jpg", warp)
    print(f"[STEP 06] Imagen warp guardada como plate_warp.jpg ({target_w}x{target_h})")

# -------------------- Preprocesado y guardado final --------------------
print("[STEP 07] Llamando a preprocess_plate para generar la imagen lista para OCR")
out_path = preprocess_plate(warp, out_path="plate_prepoc.jpg")
if out_path is None:
    print("[ERROR] Preprocesado fallido")
    raise SystemExit("Preprocesado fallido")
else:
    print("[STEP 07] Imagen preprocesada disponible en", out_path)

# -------------------- Llamada al servicio OCR (ollama) --------------------
print("[STEP 08] Preparando llamada al servicio OCR (ollama) con:", out_path)
from ollama import Client
import json

IMG = out_path #OUT  # usamos la imagen procesada

client = Client()

try:
    client = Client()
    print("[STEP 08] Cliente ollama creado. Enviando solicitud...")
    resp = client.chat(
        model="moondream",
        messages=[
            {"role": "system", "content": "You are an OCR that reads car plates and city text below."},
            {"role": "user", "content": "Read the plate and the city name from the image. Return both separated by comma. An example of the result is: 'XYZ 123 , PASTO DC' ", "images": [IMG]}
        ],
        options={
            "temperature": 0.0,
            "num_predict": 32
        }
    )
except Exception as e:
    print("[ERROR] Fallo en comunicacion con ollama:", str(e))
    raise SystemExit("Error en llamada al servicio OCR")

# validar respuesta y mostrar contenido
try:
    content = resp.get("message", {}).get("content", None)
    if content:
        print("[STEP 08] Respuesta recibida del OCR:")
        print(content)
    else:
        print("[ERROR] Respuesta del OCR sin contenido util. Respuesta cruda:")
        print(json.dumps(resp, indent=2))
except Exception as e:
    print("[ERROR] Error procesando la respuesta del OCR:", str(e))
    print("Respuesta cruda:", resp)


