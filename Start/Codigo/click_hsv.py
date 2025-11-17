import cv2, sys, numpy as np

if len(sys.argv) < 2:
    print("Uso: python click_hsv.py imagen.jpg")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    raise SystemExit("No se pudo leer la imagen.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        px = hsv[y, x]
        H, S, V = int(px[0]), int(px[1]), int(px[2])
        
        # Promedio 7x7 centrado en el clic (con bordes seguros)
        r = 3
        y1, y2 = max(0,y-r), min(hsv.shape[0], y+r+1)
        x1, x2 = max(0,x-r), min(hsv.shape[1], x+r+1)
        roi = hsv[y1:y2, x1:x2].reshape(-1,3)
        mH, mS, mV = np.mean(roi, axis=0).astype(int)
        
        print(f"Pixel ({x},{y})  H={H} S={S} V={V} | Promedio7x7 H={mH} S={mS} V={mV}")
        
        # marca visual
        cv2.circle(img, (x,y), 4, (0,0,255), -1)
        cv2.imshow("img", img)
        
        
cv2.namedWindow("img")
cv2.setMouseCallback("img", on_click)
cv2.imshow("img", img)
print("Haz CLIC sobre la PLACA amarilla (izquierdo). Pulsa 'q' para salir.")
while True:
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()       
        
        
    
       