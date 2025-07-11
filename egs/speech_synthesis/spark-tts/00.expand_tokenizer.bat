@echo off
setlocal enabledelayedexpansion

REM 获取当前脚本所在目录
set script_dir=%~dp0
REM 去掉最后的反斜杠（有些工具不喜欢）
if "%script_dir:~-1%"=="\" set script_dir=%script_dir:~0,-1%

REM 获取根目录（往上三层）
for %%i in ("%script_dir%\..\..\..") do set root_dir=%%~fi

cd /d "%root_dir%" || exit /b 1

REM 运行 Python 脚本
python -m egs.speech_synthesis.spark-tts.scripts.expand_tokenizer ^
  --base_tokenizer_path "pretrained_models/Qwen3-0.6B" ^
  --save_path "pretrained_models/spark-tts/tokenizer"

endlocal
pause
