@echo off
setlocal enabledelayedexpansion

REM 获取当前脚本所在目录
set script_dir=%~dp0
if "%script_dir:~-1%"=="\" set script_dir=%script_dir:~0,-1%

REM 获取根目录（上三级）
for %%i in ("%script_dir%\..\..\..") do set root_dir=%%~fi

cd /d "%root_dir%" || exit /b 1

REM 定义参数
set data_name=m3ed
set jsonl_file=egs/data/metadata/%data_name%.jsonl
set codec_dir=local/bicodec/%data_name%
set save_dir=local/sparktts/train_data/%data_name%
set tokenizer_path=pretrained_models/spark-tts/tokenizer

echo [INFO] data_name: %data_name%

REM 执行 Python 脚本
python -m egs.speech_synthesis.spark-tts.scripts.prepare_train ^
    --jsonl_file "%jsonl_file%" ^
    --save_dir "%save_dir%" ^
    --tokenizer_path "%tokenizer_path%" ^
    --code_dir "%codec_dir%" ^
    --gt_num 64

endlocal
pause
