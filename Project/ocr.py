# ocr.py
import json
from ollama import Client

    
    
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
