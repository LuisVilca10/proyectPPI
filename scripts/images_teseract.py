import cv2
import pytesseract
import numpy as np
import os

# Preprocesamiento global para documentos escaneados o fotografiados
def preprocesar_documento_completo(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    reescalada = cv2.resize(gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    filtrada = cv2.bilateralFilter(reescalada, 9, 75, 75)
    umbral = cv2.adaptiveThreshold(filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    invertida = cv2.bitwise_not(umbral)
    return invertida

# Extraer texto de una imagen completa preprocesada
def extraer_texto_completo(imagen, lang='spa'):
    imagen_pre = preprocesar_documento_completo(imagen)
    texto = pytesseract.image_to_string(imagen_pre, lang=lang, config='--psm 6')  # PSM 6 = bloques uniformes de texto
    texto = '\n'.join([line.strip() for line in texto.split('\n') if line.strip() != ''])
    return texto

# Procesar todas las imágenes en una carpeta y guardar los resultados .txt
def ocr_en_carpeta(ruta_carpeta, carpeta_salida="../transcripciones/images"):
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for archivo in sorted(archivos):
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"[AVISO] No se pudo cargar: {archivo}")
            continue

        print(f"[INFO] Procesando: {archivo}")
        texto = extraer_texto_completo(imagen)

        nombre_salida = os.path.splitext(archivo)[0] + "tesseract" + ".txt"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(texto)

        print(f"[✔] Transcripción guardada: {ruta_salida}")

    print(f"[✅] OCR completo. Todas las transcripciones están en '{carpeta_salida}'.")

# Ejecutar
if __name__ == "__main__":
    ocr_en_carpeta("../data/images")