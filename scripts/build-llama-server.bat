@echo off
REM Build llama-server from the TurboQuant fork of llama.cpp (Windows).
REM
REM TurboQuant adds KV cache compression (3.8-6.4x) via PolarQuant + QJL.
REM
REM Requirements:
REM   - Git, CMake >= 3.20, Visual Studio Build Tools (MSVC)
REM   - CUDA Toolkit (for GPU build)
REM
REM Usage:
REM   scripts\build-llama-server.bat
REM   scripts\build-llama-server.bat --cuda-arch 89

setlocal enabledelayedexpansion

set REPO_URL=https://github.com/TheTom/llama-cpp-turboquant.git
set BRANCH=feature/turboquant-kv-cache
set PROJECT_ROOT=%~dp0..
set BUILD_DIR=%PROJECT_ROOT%\.llama-build
set INSTALL_DIR=%PROJECT_ROOT%\llama-server
set CUDA_ARCH=

REM Parse args
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--cuda-arch" (
    set CUDA_ARCH=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:done_args

echo =============================================
echo  Building llama-server (TurboQuant fork)
echo =============================================
echo   Repo:    %REPO_URL%
echo   Branch:  %BRANCH%
echo   Output:  %INSTALL_DIR%
echo.

REM Clone or update
if exist "%BUILD_DIR%" (
    echo ^>^> Updating existing clone...
    cd /d "%BUILD_DIR%"
    git fetch origin
    git checkout %BRANCH%
    git reset --hard origin/%BRANCH%
) else (
    echo ^>^> Cloning repository...
    git clone --depth 1 --branch %BRANCH% %REPO_URL% "%BUILD_DIR%"
    cd /d "%BUILD_DIR%"
)

REM Configure CMake
echo.
echo ^>^> Configuring CMake...
set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Release

where nvcc >nul 2>nul
if %errorlevel% equ 0 (
    echo    Mode: CUDA
    set CMAKE_ARGS=%CMAKE_ARGS% -DGGML_CUDA=ON
    if not "%CUDA_ARCH%"=="" (
        set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%
    )
) else (
    echo    Mode: CPU only (install CUDA Toolkit for GPU support)
)

cmake -B build %CMAKE_ARGS%

REM Build
echo.
echo ^>^> Building (this may take several minutes)...
cmake --build build --config Release

REM Install
echo.
echo ^>^> Installing...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Find binary in various possible locations
set FOUND=0
for %%D in (
    "build\bin\Release"
    "build\bin"
    "build\Release\bin"
    "build\src\Release"
) do (
    if exist "%%~D\llama-server.exe" (
        echo    Copying from %%~D
        copy /y "%%~D\llama-server.exe" "%INSTALL_DIR%\" >nul
        copy /y "%%~D\*.dll" "%INSTALL_DIR%\" >nul 2>nul
        set FOUND=1
        goto :installed
    )
)

if %FOUND%==0 (
    echo ERROR: llama-server.exe not found after build!
    dir /s /b build\llama-server.exe 2>nul
    exit /b 1
)

:installed
echo.
echo =============================================
echo  Build complete!
echo =============================================
echo.
echo Binary: %INSTALL_DIR%\llama-server.exe
echo.
echo To use with AINow, set in .env:
echo   LLAMA_SERVER_EXE=%INSTALL_DIR%\llama-server.exe
echo.
echo TurboQuant KV cache types available:
echo   turbo2  (2-bit, most compression)
echo   turbo3  (3-bit, recommended)
echo   turbo4  (4-bit, best quality)
echo.
