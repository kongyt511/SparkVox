@echo off
rem
rem Copyright (c) 2025 SparkAudio
rem               2025 Xinsheng Wang (w.xinshawn@gmail.com)
rem
rem Licensed under the Apache License, Version 2.0
rem http://www.apache.org/licenses/LICENSE-2.0
rem
rem Example:
rem    set CUDA_VISIBLE_DEVICES=1,2,3,4,5
rem    call train.bat --config egs\codec\bicodec\config\bicodec.16k.yaml ^
rem                   --log_dir egs\codec\bicodec\results\bicodec.16k ^
rem                   --nproc_per_node 5

rem ===== Default parameters =====
set "config=egs\codec\bicodec\config\bicodec.16k.yaml"
set "log_dir=results\bicodec.16k"
set "nnodes=1"
set "nproc_per_node=-1"
set "resume=0"
set "version=null"
set "port=10086"
rem ===============================

rem Get absolute path of script
set script_dir=%~dp0
cd /d "%script_dir%"
rem Get root dir (go up 3 levels)
cd ..\..\..
set root_dir=%cd%

rem Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--config" (
    set "config=%~2"
    shift
) else if "%~1"=="--log_dir" (
    set "log_dir=%~2"
    shift
) else if "%~1"=="--nnodes" (
    set "nnodes=%~2"
    shift
) else if "%~1"=="--nproc_per_node" (
    set "nproc_per_node=%~2"
    shift
) else if "%~1"=="--resume" (
    set "resume=%~2"
    shift
) else if "%~1"=="--version" (
    set "version=%~2"
    shift
) else if "%~1"=="--port" (
    set "port=%~2"
    shift
)
shift
goto parse_args
:args_done

rem Make log_dir absolute if not already
echo %log_dir% | findstr /b /c:"\" >nul
if errorlevel 1 (
    set "log_dir=%root_dir%\%log_dir%"
)

rem Create log_dir if needed
if not exist "%log_dir%" (
    mkdir "%log_dir%"
    echo Log directory created: %log_dir%
)

rem Get timestamp
for /f "tokens=1-4 delims=/- " %%a in ('wmic os get LocalDateTime ^| find "."') do (
    set datetime=%%a
)
set tag=%datetime:~0,8%_%datetime:~8,6%

rem Write train.bat in log_dir
set trainbat=%log_dir%\%tag%_train.bat

(
echo @echo off
echo cd /d "%root_dir%"
echo python -m bins.train_pl ^
 --config %config% ^
 --log_dir "%log_dir%" ^
 --resume %resume% ^
 --nnodes %nnodes% ^
 --nproc_per_node %nproc_per_node% ^
 --version %version% ^
 --date %tag%
) > "%trainbat%"

echo Run batch saved to: %trainbat%
echo Executing: %trainbat%

call "%trainbat%"
