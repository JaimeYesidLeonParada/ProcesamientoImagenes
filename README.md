# 🚗 Detección de Placas Vehiculares en el Borde para Movilidad Urbana

Este proyecto implementa un sistema de **reconocimiento automático de placas vehiculares (ALPR/ANPR)** con procesamiento en el borde, diseñado para funcionar en tiempo real sobre hardware embebido como **Raspberry Pi**.  
El sistema integra un pipeline de procesamiento de imágenes en **Python + OpenCV** y un módulo OCR basado en **Ollama**, logrando un prototipo funcional para aplicaciones de movilidad urbana.

---

## 📑 Resumen del proyecto

- **Objetivo:** Detectar y reconocer placas vehiculares en tiempo real sin depender de la nube, mejorando latencia, costos y privacidad.
- **Pipeline:** Captura → Segmentación por color (HSV) → Detección de contornos → Rectificación geométrica (warp) → Preprocesamiento (CLAHE, filtros, sharpen) → OCR con Ollama → Registro de resultados.
- **Dataset:** Más de 100 imágenes de placas recolectadas y procesadas.  [👉 Base de datos de Imagenes](https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/tree/main/Project/data)
- **Evaluación:** Se seleccionaron 38 imágenes representativas para análisis detallado.
- **Resultados:**  
  - 42% de detecciones completas (placa + ciudad).  
  - Puntuación total: 87/114 (76%).  
  - Tiempo promedio de procesamiento: **54.860 segundos por imagen**.

---

## 🛠️ Arquitectura del sistema

1. **Creación de máscara (HSV + morfología):**  
   Conversión de BGR a HSV y aplicación de umbrales para aislar el color predominante de las placas (H=17–27, S=160–255, V=190–255).

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/Codigo/overlay_hsv.jpg" width="300"/>
</p>

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/Codigo/debug_mask_raw.png" width="300"/>
</p>

3. **Detección de contornos:**  
   Selección del contorno de mayor área y cálculo de la caja mínima para localizar la placa.

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/Codigo/step2_contour.jpg" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/Codigo/step3_contour_fixed.jpg" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/Codigo/step4_minrect.jpg" width="500"/>
</p>

5. **Rectificación geométrica (warp):**  
   Homografía para normalizar la perspectiva y obtener una placa alineada.

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Project/debug_points.jpg" width="500"/>
</p>

7. **Preprocesamiento:**  
   - Escala de grises  
   - CLAHE (contraste adaptativo)  
   - Filtro bilateral (reducción de ruido)  
   - Sharpen (realce de bordes)

  <p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/CodigoFinal/plate_warp.jpg" width="300"/>
</p>

  <p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Start/CodigoFinal/plate_prepoc.jpg" width="300"/>
</p>


8. **OCR con Ollama:**  
   Uso del modelo multimodal `moondream` con un prompt especializado para leer placa y ciudad.  
   Ejemplo de salida:

   XYZ 123 , PASTO DC

   ```python
    from ollama import Client
    import json
    
    IMG = "plate_warp.jpg"   # ? tu imagen preprocesada
        
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

9. **Diagrama de Flujo del sistema:**

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Project/grafica_flujoinferencia.png" width="500"/>
</p>

10. **Registro de resultados:**  
Los resultados se guardan en un archivo CSV con:

[Ver tabla completa](https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Project/resultados.csv)

---

## 📊 Evaluación del sistema

- **Metodología de puntuación:**  
- 3 puntos → placa y ciudad correctas  
- 2 puntos → placa parcial + ciudad correcta  
- 1 punto → solo ciudad detectada  

- **Resumen:**  
- Imágenes evaluadas: 38  
- Detecciones correctas: 16 (42%)  
- Puntuación total: 87/114 (76%)

- **Visualizaciones:**  
- Distribución de puntuaciones (1, 2 y 3 puntos).

<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Project/grafica_puntuacion.png" width="500"/>
</p>
  
- Número de detecciones completas vs parciales.  
<p align="center">
  <img src="https://github.com/JaimeYesidLeonParada/ProcesamientoImagenes/blob/main/Project/grafica_rendimiento.png" width="500"/>
</p>

---

## 🚀 Ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tuusuario/tu-repo.git
   cd tu-repo

2. Instalar dependencias:
   pip install -r requirements.txt

3. Ejecutar el pipeline:
   python main.py data/placas
   
4. Revisar resultados en:
     - Consola (OCR bruto y limpio, tiempos).
    - Archivo resultados.csv.

## 📈 Conclusiones
- El sistema demostró ser funcional en condiciones controladas, con un 42% de detecciones completas.
- Ollama ofreció mayor robustez que Tesseract, aunque con tiempos de procesamiento elevados.
- El pipeline de procesamiento de imágenes fue clave para mejorar la calidad de entrada al OCR.

## 🔮 Trabajo futuro
- Optimizar preprocesamiento para condiciones nocturnas.
- Ampliar dataset con diferentes tipos de placas.
- Integrar aceleradores de hardware y optimización de modelos.
- Validar en escenarios urbanos reales con video en tiempo real.

## 👤 Autor
Jaime Yesid Leon Parada
Pontificia Universidad Javeriana
Procesamiento de Imágenes y Video
📧 leon-jaime@javeriana.edu.co

