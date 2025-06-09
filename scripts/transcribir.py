import os
import whisper
import shutil
import torch
import sys
from tqdm import tqdm

#verifica si esta instalado ffmpeg
def verificar_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("❌ FFmpeg no está instalado o no está en el PATH.")
        sys.exit(1)

# Cargar el modelo Whisper ("base", "small", "medium" o "large")
def cargar_modelo(nombre="medium"):
    if not torch.cuda.is_available():
        print("⚠️ Advertencia: GPU no disponible, se usará CPU.")
        return whisper.load_model(nombre)
    return whisper.load_model(nombre).to("cuda")

# Transcribe audio en español a texto con whisper
def transcribir_audio(modelo, input_path):
    return modelo.transcribe(input_path, language="es")["text"]

#Guarda el texto transcrito en un archivo .txt
def guardar_transcripcion(output_dir, filename, texto):
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"✅ Guardado: {output_filename}")

# valida carpeta de salida y que tipo de archivos son permitidos
def procesar_audios(audio_dir, output_dir, modelo):
    os.makedirs(output_dir, exist_ok=True)
    EXTENSIONES_VALIDAS = (".mp3", ".wav", ".m4a", ".flac")

    for filename in tqdm(os.listdir(audio_dir), desc="Procesando audios"):
        if filename.lower().endswith(EXTENSIONES_VALIDAS):
            input_path = os.path.join(audio_dir, filename)
            print(f"🔊 Procesando: {filename}")
            try:
                texto = transcribir_audio(modelo, input_path)
                guardar_transcripcion(output_dir, filename, texto)
            except Exception as e:
                print(f"⚠️ Error en {filename}: {e}")

if __name__ == "__main__":
    verificar_ffmpeg()
    modelo = cargar_modelo() #podemos especifcar que modelo usar "cargar_modelo("base")"
    audio_dir = "../data/audios"
    output_dir = "../data/transcripciones"
    procesar_audios(audio_dir, output_dir, modelo)
    print("🎉 Transcripciones completadas.")
