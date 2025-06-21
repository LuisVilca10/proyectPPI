import boto3
import cv2
import numpy as np
import os

# --- Preprocesamiento con OpenCV ---
def preprocesar_documento_completo(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    reescalada = cv2.resize(gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    filtrada = cv2.bilateralFilter(reescalada, 9, 75, 75)
    umbral = cv2.adaptiveThreshold(filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    invertida = cv2.bitwise_not(umbral)
    return invertida

# --- Guardar texto en archivo ---
def guardar_texto(nombre_archivo, texto, carpeta="../transcripciones/images"):
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre_archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"[✔] Texto guardado en: {ruta}")

# --- OCR usando AWS Textract (solo texto plano) ---
def ocr_textract_texto_plano(ruta_imagen):
    textract = boto3.client('textract')

    imagen = cv2.imread(ruta_imagen)
    imagen_pre = preprocesar_documento_completo(imagen)

    _, buffer = cv2.imencode('.jpeg', imagen_pre)
    document_bytes = buffer.tobytes()

    response = textract.detect_document_text(Document={'Bytes': document_bytes})

    lineas = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
    texto = "\n".join(lineas)
    return texto

# --- Ejecutar OCR ---
if __name__ == "__main__":
    ruta_imagen = "../data/images/requirements.jpeg"
    texto_extraido = ocr_textract_texto_plano(ruta_imagen)
    guardar_texto("requirements_opencv_textract.txt", texto_extraido)
