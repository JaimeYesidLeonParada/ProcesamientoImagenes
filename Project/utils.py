# utils.py
import cv2
import numpy as np

# create_mask: convierte a HSV, crea mascara y aplica morfologia

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
    
# find_largest_contour: busca contornos y devuelve el mayor y su caja

def find_largest_contour(mask, kernel):
    print("[FNC find_largest_contour] Aplicando CLOSE adicional y buscando contornos")
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[FNC find_largest_contour] No se encontraron contornos")
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    print(f"[FNC find_largest_contour] Contorno mayor encontrado. Area = {int(area)}")

    # calcular caja minima
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)

    w, h = rect[1]
    aspect = max(w, h) / (min(w, h) + 1e-5)

    if aspect > 4.0 or aspect < 1.5:
        # fallback: boundingRect
        x, y, bw, bh = cv2.boundingRect(cnt)
        box = np.array([
            [x, y],
            [x+bw, y],
            [x+bw, y+bh],
            [x, y+bh]
        ], dtype=np.float32)
        print(f"[FNC find_largest_contour] Aspecto raro ({aspect:.2f}). Usando boundingRect.")
    else:
        print(f"[FNC find_largest_contour] Aspecto aceptable ({aspect:.2f}). Usando minAreaRect.")

    return cnt, box

    
def expand_box(src_points, expand_px=5):
    # src_points: array de 4 puntos (tl, tr, br, bl)
    # expandimos cada punto hacia afuera
    expanded = src_points.copy()
    expanded[0][0] -= expand_px; expanded[0][1] -= expand_px  # tl
    expanded[1][0] += expand_px; expanded[1][1] -= expand_px  # tr
    expanded[2][0] += expand_px; expanded[2][1] += expand_px  # br
    expanded[3][0] -= expand_px; expanded[3][1] += expand_px  # bl
    return expanded
    
# get_warp_from_box: ordena puntos, calcula homografia y genera warp
def get_warp_from_box(box, image, target_h=240, min_w=120):
    # ordenar puntos como tl, tr, br, bl
    print("[FNC get_warp_from_box] Ordenando puntos fuente")
    
    #s = box.sum(axis=1)
    #diff = np.diff(box, axis=1).reshape(-1)
    #tl = box[np.argmin(s)]
    #br = box[np.argmax(s)]
    #tr = box[np.argmin(diff)]
    #bl = box[np.argmax(diff)]
    #src = np.array([tl, tr, br, bl], dtype=np.float32)

    src = order_points(box)
    src = expand_box(src, expand_px=8)
    print("[FNC get_warp_from_box] Puntos fuente ordenados:", src.tolist())

    debug_points = image.copy()
    for (x, y) in src.astype(int):
    	cv2.circle(debug_points, (x, y), 5, (0, 255, 0), -1)
    cv2.imwrite("debug_points.jpg", debug_points)
    print("[DEBUG] Puntos fuente guardados en debug_points.jpg")

    src = expand_box(src, expand_px=8)  # anade margen de 8 pixeles
    print("[FNC get_warp_from_box] Puntos fuente ordenados:", src.tolist())

    # calcular ancho y alto del rectificado
    #wA = np.linalg.norm(br - bl)
    #wB = np.linalg.norm(tr - tl)
    #hA = np.linalg.norm(tr - br)
    #hB = np.linalg.norm(tl - bl)
    
    wA = np.linalg.norm(src[2] - src[3])  # br - bl
    wB = np.linalg.norm(src[1] - src[0])  # tr - tl
    hA = np.linalg.norm(src[1] - src[2])  # tr - br
    hB = np.linalg.norm(src[0] - src[3])  # tl - bl
    
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
    
 # preprocess_plate: CLAHE, bilateral, blur+sharpen y guardado
 
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

def order_points(pts):
    # pts: array de 4 puntos (x,y)
    rect = np.zeros((4, 2), dtype="float32")

    # ordenar por x (izquierda vs derecha)
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left = x_sorted[:2]
    right = x_sorted[2:]

    # ordenar por y dentro de cada grupo
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]

    # asignar: tl, bl, tr, br
    rect[0] = left[0]   # top-left
    rect[1] = right[0]  # top-right
    rect[2] = right[1]  # bottom-right
    rect[3] = left[1]   # bottom-left

    return rect
