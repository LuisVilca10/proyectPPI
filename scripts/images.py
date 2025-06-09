import cv2
import pytesseract
import numpy as np

imagen = cv2.imread('../data/images/imagen.png')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calcular anchos y altos
anchos = []
altos = []
for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    anchos.append(w)
    altos.append(h)

p_anchos = np.percentile(anchos, 40)
p_altos = np.percentile(altos, 40)

bloques_texto = []
for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if w > p_anchos and h > p_altos and area > 500:  # Agregamos área mínima
        roi = imagen[y:y+h, x:x+w]
        bloques_texto.append((x, y, roi))

bloques_texto.sort(key=lambda b: b[1])

texto_final = ""
for i, (x, y, roi) in enumerate(bloques_texto):
    texto_bloque = pytesseract.image_to_string(roi, lang='spa')
    texto_final += texto_bloque.strip() + "\n"

# Limpieza de saltos de línea y espacios en blanco
texto_final = '\n'.join([line.strip() for line in texto_final.split('\n') if line.strip() != ''])

print("=== Texto final limpio y estructurado ===")
print(texto_final)
