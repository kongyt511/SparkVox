@echo off
setlocal enabledelayedexpansion

REM 获取当前脚本所在目录
set script_dir=%~dp0
if "%script_dir:~-1%"=="\" set script_dir=%script_dir:~0,-1%

REM 获取根目录（往上三层）
for %%i in ("%script_dir%\..\..\..") do set root_dir=%%~fi

cd /d "%root_dir%" || exit /b 1

REM 配置参数
set batch_size=32
set data_name=m3ed
set jsonlfile=egs/data/metadata/m3ed.jsonl
set save_dir=local/%data_name%

set config_path=/aifs4su/xinshengwang/code/SparkAudio/SparkVox/egs/codec/bicodec/results/bicodec.24k/20250420_014312/config.yaml
set ckpt_path=/aifs4su/xinshengwang/code/VoxSphere/egs/recipes/librispeech/ssl2wav/results/20241202.ema.wav2vec.lmix.8192.spkFSQ.dualEncoder.fvq/ckpt/800000.pt
set data_root=%root_dir%/egs/data/audios

echo [INFO] 根目录: %root_dir%
echo [INFO] 数据目录: %data_root%

REM 运行 Python 脚本
python -m sparkvox.tools.tokenizer.audio_tokenizer.bicodec.extract_codes ^
  --jsonlfile "%jsonlfile%" ^
  --data_root "%data_root%" ^
  --config_path "%config_path%" ^
  --ckpt_path "%ckpt_path%" ^
  --save_dir "%save_dir%" ^
  --batch_size %batch_size%

endlocal
pause
