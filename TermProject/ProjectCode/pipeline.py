
import whisper, torch, subprocess
print("Whisper", whisper.__version__, "| CUDA:", torch.cuda.is_available())
subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL)
print("ffmpeg OK")
