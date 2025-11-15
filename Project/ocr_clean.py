# ocr_clean.py
import re

def clean_ocr_text(raw):
    """
    Entrada: texto crudo del OCR (ej: 'JUNI 540, BOGOTA DC')
    Salida: (plate_fixed, rest_text)
      - plate_fixed: 'ABC 123'
      - rest_text: lo que sigue (ej: 'BOGOTA DC')
    """
    if not raw or not isinstance(raw, str):
        return '', ''

    s = raw.strip().upper()
    # conservar solo letras, numeros, espacios y comas
    s = re.sub(r'[^A-Z0-9 ,]', '', s)

    # separar por coma para aislar ciudad
    parts = [p.strip() for p in s.split(',') if p.strip()]
    plate_part = parts[0] if parts else s
    rest_text = parts[1] if len(parts) > 1 else ''

    # extraer letras y numeros
    letters = re.findall(r'[A-Z]', plate_part)
    digits = re.findall(r'[0-9]', plate_part)

    # recortar a maximo 3
    letters = letters[:3]
    digits = digits[:3]

    if len(letters) == 3 and len(digits) == 3:
        plate_fixed = f"{''.join(letters)} {''.join(digits)}"
    else:
        # fallback: devolver lo que haya
        plate_fixed = plate_part

    return plate_fixed, rest_text
