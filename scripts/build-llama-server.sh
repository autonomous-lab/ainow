#!/usr/bin/env bash
#
# Build llama-server from the TurboQuant fork of llama.cpp.
#
# TurboQuant adds KV cache compression (3.8-6.4x) via PolarQuant + QJL,
# enabling much larger context windows and bigger models in the same VRAM.
#
# Usage:
#   ./scripts/build-llama-server.sh              # auto-detect GPU
#   ./scripts/build-llama-server.sh --cuda-arch 89  # specific arch (e.g. RTX 4090)
#   ./scripts/build-llama-server.sh --cpu           # CPU-only build
#
# Requirements:
#   - Git, CMake >= 3.20, C/C++ compiler (MSVC on Windows, GCC/Clang on Linux)
#   - CUDA Toolkit (for GPU build)
#
# Output:
#   Build artifacts are placed in ./llama-server/ relative to this script's
#   parent directory (the AINow project root).

set -euo pipefail

REPO_URL="https://github.com/TheTom/llama-cpp-turboquant.git"
BRANCH="feature/turboquant-kv-cache"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/.llama-build"
INSTALL_DIR="$PROJECT_ROOT/llama-server"

CUDA_ARCH=""
CPU_ONLY=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --cpu) CPU_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--cuda-arch <arch>] [--cpu]"
            echo "  --cuda-arch <arch>  CUDA compute capability (e.g. 89 for RTX 4090)"
            echo "  --cpu               Build without GPU acceleration"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo " Building llama-server (TurboQuant fork)"
echo "============================================="
echo "  Repo:    $REPO_URL"
echo "  Branch:  $BRANCH"
echo "  Output:  $INSTALL_DIR"
echo ""

# Clone or update
if [ -d "$BUILD_DIR" ]; then
    echo ">> Updating existing clone..."
    cd "$BUILD_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    echo ">> Cloning repository..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Configure CMake
echo ""
echo ">> Configuring CMake..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

if [ "$CPU_ONLY" = true ]; then
    echo "   Mode: CPU only"
elif command -v nvcc &>/dev/null; then
    echo "   Mode: CUDA (nvcc found)"
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
    if [ -n "$CUDA_ARCH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    fi
elif command -v hipcc &>/dev/null; then
    echo "   Mode: ROCm/HIP (hipcc found)"
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
else
    echo "   Mode: CPU (no CUDA/HIP detected)"
    echo "   WARNING: No GPU acceleration. Install CUDA Toolkit for GPU support."
fi

cmake -B build $CMAKE_ARGS

# Build
echo ""
echo ">> Building (this may take a few minutes)..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build build --config Release -j "$NPROC"

# Install
echo ""
echo ">> Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# Find and copy the binary
if [ -f "build/bin/Release/llama-server.exe" ]; then
    cp build/bin/Release/llama-server.exe "$INSTALL_DIR/"
    # Copy DLLs (CUDA runtime, etc.)
    find build/bin/Release/ -name "*.dll" -exec cp {} "$INSTALL_DIR/" \; 2>/dev/null || true
elif [ -f "build/bin/llama-server.exe" ]; then
    cp build/bin/llama-server.exe "$INSTALL_DIR/"
    find build/bin/ -name "*.dll" -exec cp {} "$INSTALL_DIR/" \; 2>/dev/null || true
elif [ -f "build/bin/llama-server" ]; then
    cp build/bin/llama-server "$INSTALL_DIR/"
    find build/bin/ -name "*.so" -exec cp {} "$INSTALL_DIR/" \; 2>/dev/null || true
elif [ -f "build/Release/bin/llama-server.exe" ]; then
    cp build/Release/bin/llama-server.exe "$INSTALL_DIR/"
    find build/Release/bin/ -name "*.dll" -exec cp {} "$INSTALL_DIR/" \; 2>/dev/null || true
else
    echo "ERROR: llama-server binary not found after build!"
    echo "Searching build directory..."
    find build -name "llama-server*" -type f 2>/dev/null
    exit 1
fi

echo ""
echo "============================================="
echo " Build complete!"
echo "============================================="
echo ""
echo "Binary: $INSTALL_DIR/llama-server$([ -f '$INSTALL_DIR/llama-server.exe' ] && echo '.exe' || echo '')"
echo ""
echo "To use with AINow, set in .env:"
echo "  LLAMA_SERVER_EXE=$INSTALL_DIR/llama-server$([ -f '$INSTALL_DIR/llama-server.exe' ] && echo '.exe' || echo '')"
echo ""
echo "TurboQuant KV cache types available:"
echo "  turbo2  (2-bit, most compression, slight quality loss)"
echo "  turbo3  (3-bit, recommended balance)"
echo "  turbo4  (4-bit, least compression, best quality)"
echo ""
