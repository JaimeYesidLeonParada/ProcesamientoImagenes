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
print("[STEP 03] Convirtiendo a HSV y generando mascara")

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower = np.array([HMIN, SMIN, VMIN], dtype=np.uint8)
upper = np.array([HMAX, SMAX, VMAX], dtype=np.uint8)

h, w, _  = hsv.shape
k = max(3, min(15, w // 100)) # Escala: 3..15 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
mask = cv2.inRange(hsv, lower, upper)
print(f"[STEP 03] Mascara inicial creada. Pixeles blancos: {int(mask.sum() / 255)} / {w*h}")

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
print("[STEP 03] Aplicadas operaciones morfologicas: OPEN, CLOSE")

# -------------------- Overlay y blend --------------------
print("[STEP 04] Creando overlay y blend para debug")

# crear overlay marcando pixeles detectados en rojo
overlay = bgr.copy()
overlay[mask > 0] = (0, 0, 255)

# mezclar original y overlay para visualizacion

blend = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
print("[STEP 04] Overlay y blend creados")


# -------------------- Deteccion de contornos y seleccion --------------------
print("[STEP 05] Buscando contornos en la mascara")

# cerrar small gaps antes de buscar contornos

mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("[ERROR] No se encontraron contornos en la mascara")
    raise SystemExit("No se encontraron contornos.")

# elegir el contorno de mayor area
cnt = max(contours, key=cv2.contourArea)
area = cv2.contourArea(cnt)
print(f"[STEP 05] Contorno seleccionado. Area = {int(area)} pixeles")

# obtener rectangulo minimo y puntos de caja
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect).astype(np.float32)
print("[STEP 05] Caja minima obtenida. Coordenadas de la caja:", box.tolist())


# -------------------- Ordenar puntos y calcular homografia --------------------
print("[STEP 06] Ordenando puntos y calculando homografia")

# ordenar la caja como tl, tr, br, bl
s = box.sum(axis=1)
diff = np.diff(box, axis=1).reshape(-1)
tl = box[np.argmin(s)]
br = box[np.argmax(s)]
tr = box[np.argmin(diff)]
bl = box[np.argmax(diff)]
src = np.array([tl, tr, br, bl], dtype=np.float32)
print("[STEP 06] Puntos fuente ordenados:", src.tolist())

# calcular ancho y alto del rectificado
wA = np.linalg.norm(br - bl)
wB = np.linalg.norm(tr - tl)
hA = np.linalg.norm(tr - br)
hB = np.linalg.norm(tl - bl)
maxW = max(wA, wB)
maxH = max(hA, hB)
target_h = 240
aspect   = maxW / maxH if maxH > 0 else 4.0
target_w = int(max(120, target_h * aspect)) 
print(f"[STEP 06] Tamano destino: {target_w}x{target_h} (aspecto {aspect:.2f})")

# puntos destino y homografia
dst = np.array([[0, 0], [target_w -1, 0], [target_w -1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src, dst)
print("[STEP 06] Matriz de transformacion calculada")

# aplicar warp
warp = cv2.warpPerspective(bgr, M, (target_w, target_h), flags= cv2.INTER_LINEAR)

if warp is None or warp.size == 0:
    print("[ERROR] Fallo al generar la imagen warp")
    raise SystemExit("No se pudo crear warp")
else:
    print("[STEP 06] Warp creado correctamente")

# -------------------- Preprocesado y guardado final --------------------
print("[STEP 07] Iniciando preprocesado de la imagen warp")


OUT = "plate_prepoc.jpg"
img = warp

if img is None:
    print("[ERROR] No se encontro la imagen con el wrap")
    raise SystemExit("No existe la imagen warp")

print("[STEP 07] Conversion a gris")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("[STEP 07] Aplicando CLAHE")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

print("[STEP 07] Aplicando filtro bilateral para reducir ruido preservando bordes")
gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

print("[STEP 07] Aplicando desenfoque y realce (sharpen)")
blur  = cv2.GaussianBlur(gray, (0,0), 1.0)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

print("[STEP 07] Convirtiendo a BGR y guardando:", OUT)
out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
cv2.imwrite(OUT, out)

print("[STEP 07] Guardado completado:", OUT)

# -------------------- Llamada al servicio OCR (ollama) --------------------
print("[STEP 08] Preparando llamada al servicio OCR (ollama) con:", OUT)
from ollama import Client
import json

IMG = OUT  # usamos la imagen procesada

client = Client()

try:
    client = Client()
    print("[STEP 08] Cliente ollama creado. Enviando solicitud...")
    resp = client.chat(
        model="moondream",
        messages=[
            {"role": "system", "content": "You are an OCR that reads car plates and city text below."},
            {"role": "user", "content": "Read the plate and the city name from the image. Return both separated by comma. Example: JNU540, BOGOTA DC", "images": [IMG]}
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


