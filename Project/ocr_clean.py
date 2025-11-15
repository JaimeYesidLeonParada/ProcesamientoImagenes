# ocr_clean.py
import re

def clean_ocr_text(raw):
    if not raw or not isinstance(raw, str):
        return '', ''

    s = raw.strip().upper()
    s = re.sub(r'[^A-Z0-9 ,]', '', s)

    # separar por coma primero
    parts = [p.strip() for p in s.split(',') if p.strip()]
    plate_part = parts[0] if parts else s
    rest_text = ', '.join(parts[1:]) if len(parts) > 1 else ''

    # buscar letras y numeros
    letters = re.findall(r'[A-Z]', plate_part)
    digits = re.findall(r'[0-9]', plate_part)

    letters = letters[:3]
    digits = digits[:3]

    if len(letters) == 3 and len(digits) == 3:
        plate_fixed = f"{''.join(letters)} {''.join(digits)}"
    else:
        plate_fixed = plate_part

    # si no hay coma, tomar lo que sigue como ciudad
    if not rest_text:
        # todo lo que quede despues de los digitos
        m = re.search(r'[0-9]{3}(.*)', s)
        if m:
            rest_text = m.group(1).strip().lstrip(',').strip()

    return plate_fixed, rest_text

