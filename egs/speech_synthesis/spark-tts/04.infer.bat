@echo off
setlocal enabledelayedexpansion

REM 获取当前脚本所在目录
set script_dir=%~dp0
if "%script_dir:~-1%"=="\" set script_dir=%script_dir:~0,-1%

REM 获取根目录（上三级）
for %%i in ("%script_dir%\..\..\..") do set root_dir=%%~fi

REM 参数
set audio_tokenizer_config=egs/speech_synthesis/spark-tts/config/audio_tokenizer/bicodec.yaml
set ckpt=egs/speech_synthesis/spark-tts/results/sparktts_qwen3-0.6b/20250609_145031/ckpt/epoch-2.ckpt
set save_dir=local/sparktts/infer/sparktts_qwen3-0.6b/20250609_145031
set wav_path=egs/data/audios/m3ed/m3ed_Neutral_0000022926.wav
set text=靠你这张脸买东西能打折。
set prompt_text=我怕你万一累出个三长两短来。
set device=0

REM 切换到根目录
cd /d "%root_dir%" || exit /b 1

REM 执行推理
python -m sparkvox.models.speech_synthesis.sparktts.inference_single ^
    --ckpt "%ckpt%" ^
    --audio_tokenizer_config "%audio_tokenizer_config%" ^
    --save_dir "%save_dir%" ^
    --wav_path "%wav_path%" ^
    --text "%text%" ^
    --device %device% ^
    --prompt_text "%prompt_text%"

endlocal
pause
