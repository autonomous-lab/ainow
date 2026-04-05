@echo off
REM Optimal config for 16GB VRAM: Qwen 9B UD-Q4_K_XL (~6GB) + Whisper large-v3 (~3GB)
set WHISPER_MODEL=large-v3
python main.py --model 9b %*
