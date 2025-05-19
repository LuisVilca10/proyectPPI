import os
import whisper
import shutil
import sys
# Verificar si ffmpeg está disponible
if not shutil.which("ffmpeg"):
    print("❌ FFmpeg no está instalado o no está en el PATH. El programa no puede continuar.")
    sys.exit(1)

# Cargar el modelo Whisper (puedes cambiar "base" por "small", "medium" o "large" si tienes más recursos)
model = whisper.load_model("base")

# Definir rutas de entrada y salida
audio_dir = "../data/audios"
output_dir = "../data/transcripciones"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer todos los archivos en la carpeta de audios
for filename in os.listdir(audio_dir):
    if filename.lower().endswith(".mp3"):
        input_path = os.path.join(audio_dir, filename)
        print(f"🔊 Procesando: {filename}")

        # Transcribir con Whisper
        result = model.transcribe(input_path, language="es")
        texto = result["text"]

        # Guardar transcripción como .txt con mismo nombre
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(texto)
        
        print(f"✅ Guardado: {output_filename}")

print("🎉 Transcripciones completadas.")
