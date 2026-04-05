@echo off
REM Comfort config for 16GB VRAM: Qwen 9B UD-Q4_K_XL (~6GB) + Whisper medium (~1.5GB) + 256K context
set WHISPER_MODEL=medium
python main.py --model 9b %*
