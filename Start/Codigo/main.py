import cv2
import numpy as np

IMG_PATH = "/home/ingen/Documents/ProcesamientodeImagenes/ProyectoFinal/ImagenesPlacas/placa1.jpg"

# Umbrales
# H en OpenCV va 0..179; S,V en 0..255
HMIN, HMAX = 17, 27
SMIN, SMAX = 160, 255
VMIN, VMAX = 190, 255

img = cv2.imread(IMG_PATH)
#resized = cv2.resize(img, (350, 250), interpolation=cv2.INTER_CUBIC)
#bgr = resized.copy()

#bgr = cv2.imread(IMG_PATH)
bgr = cv2.resize(img, (340, 250), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("imagen_rezie.jpg", bgr)
    


if bgr is None:
    raise SystemExit("No se pudo leer la imagen.")

# --- Convertir a HSV ---
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower = np.array([HMIN, SMIN, VMIN], dtype=np.uint8)
upper = np.array([HMAX, SMAX, VMAX], dtype=np.uint8)

h, w, _  = hsv.shape
k = max(3, min(15, w // 100)) # Escala: 3..15 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
mask = cv2.inRange(hsv, lower, upper)
cv2.imwrite("debug_mask_raw.png", mask)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite("debug_mask_open.png", mask)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

overlay = bgr.copy()
overlay[mask > 0] = (0, 0, 255)
blend = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

cv2.imwrite("mask_hsv.png", mask)
cv2.imwrite("overlay_hsv.jpg", blend)

print("OK guardado: mask_hsv.png y overlay_hasv.jpg")

mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite("debug_mask_close.png", mask_closed)

contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

debug_contours = bgr.copy()
cv2.drawContours(debug_contours, contours, -1, (0,255,0), 2)
cv2.imwrite("debug_contours.jpg", debug_contours)


if not contours:
    raise SystemExit("No se encontraron contornos.")

cnt = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect).astype(np.int32)
debug_min = bgr.copy()
cv2.drawContours(debug_min, [box], -1, (0,0,255), 2)
cv2.imwrite("debug_box_min.jpg", debug_min)



# boundingRect
x,y,w,h = cv2.boundingRect(cnt)
box_bound = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.int32)
debug_bound = bgr.copy()
cv2.drawContours(debug_bound, [box_bound], 0, (255,0,0), 2)
cv2.imwrite("debug_box_bound.jpg", debug_bound)





s = box.sum(axis=1)
diff = np.diff(box, axis=1).reshape(-1)
tl = box[np.argmin(s)]
br = box[np.argmax(s)]
tr = box[np.argmin(diff)]
bl = box[np.argmax(diff)]
src = np.array([tl, tr, br, bl], dtype=np.float32)

wA = np.linalg.norm(br - bl)
wB = np.linalg.norm(tr - tl)
hA = np.linalg.norm(tr - br)
hB = np.linalg.norm(tl - bl)
maxW = max(wA, wB)
maxH = max(hA, hB)
target_h = 240
aspect   = maxW / maxH if maxH > 0 else 4.0
target_w = int(max(120, target_h * aspect)) 

dst = np.array([[0, 0], [target_w -1, 0], [target_w -1, target_h - 1], [0, target_h - 1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(bgr, M, (target_w, target_h), flags= cv2.INTER_LINEAR)

cv2.imwrite("plate_warp.jpg", warp)
print("[OK] guardado plate_warp.jpg")

IN = "plate_warp.jpg"
OUT = "plate_prepoc.jpg"

img = cv2.imread(IN)
if img is None:
    raise SystemExit("No existe el archivo plate_wrap.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

blur  = cv2.GaussianBlur(gray, (0,0), 1.0)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
cv2.imwrite(OUT, out)

print("[OK] guardado", OUT)

#-------------------
from ollama import Client
import json

IMG = "plate_warp.jpg"   # ? tu imagen preprocesada
#client = Client(host="http://localhost:11434")

client = Client()

resp = client.chat(
    model="moondream",
    messages=[
        {"role":"system","content":"You are an OCR that reads car plates and city text below."},
        {"role":"user","content":"Read the plate and the city name from the image. Return both separated by comma. Example: ABC540, BOGOTA DC","images":["plate_prepoc.jpg"]}
    ],
    options={
        "temperature": 0.0,
        "num_predict": 32
    }
)
print(resp["message"]["content"])
