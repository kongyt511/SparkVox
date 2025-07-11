@echo off
setlocal enabledelayedexpansion

REM 环境变量
set LOGLEVEL=INFO
set NCCL_DEBUG=INFO
set PYTHONWARNINGS=ignore

REM 获取当前脚本所在目录
set script_dir=%~dp0
if "%script_dir:~-1%"=="\" set script_dir=%script_dir:~0,-1%

REM 获取根目录（往上三级）
for %%i in ("%script_dir%\..\..\..") do set root_dir=%%~fi

REM 默认参数
set config=egs/speech_synthesis/spark-tts/config/spark-tts_qwen3-0.6b.yaml
set log_dir=%script_dir%\results\sparktts_qwen3-0.6b
set nnodes=1
set nproc_per_node=-1
set num_workers=8
set accumluate=12
set resume=0
set version=null
set port=10086

REM 切换目录
cd /d "%root_dir%" || exit /b 1

REM 确保 log_dir 是绝对路径
echo %log_dir% | findstr /b /c:"\" >nul
if errorlevel 1 (
    set log_dir=%root_dir%\%log_dir%
)

REM 创建 log_dir
if "%resume%"=="0" (
    if not exist "%log_dir%" (
        mkdir "%log_dir%"
        echo [INFO] Log directory created: %log_dir%
    )
)

REM 时间戳
for /f %%t in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set tag=%%t

if "%version%"=="null" (
    set version=%tag%
)

REM 生成训练脚本
set train_script=%log_dir%\%tag%_train.bat
(
    echo @echo off
    echo cd /d "%root_dir%" || exit /b 1
    echo python -m bins.train_pl ^
        --config %config% ^
        --log_dir "%log_dir%" ^
        --resume %resume% ^
        --nnodes %nnodes% ^
        --nproc_per_node %nproc_per_node% ^
        --accumluate %accumluate% ^
        --version %version% ^
        --date %tag%
) > "%train_script%"

echo [INFO] Run script saved to %train_script%

echo [INFO] Execute %train_script%

call "%train_script%"

endlocal
pause
