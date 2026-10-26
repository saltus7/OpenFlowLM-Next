<#
.SYNOPSIS
    Builds and launches OpenFlowLM-Next with the ironenv toolchain.

.DESCRIPTION
    This script automates the kernel export, build, and launch process for OpenFlowLM-Next.
    It requires the specific 'ironenv' virtual environment to be active before running.

    ------------------------------------------------------------------
    ENVIRONMENT ARCHITECTURE (One-Stop Documentation)
    ------------------------------------------------------------------
    There are two separate Python environments on this system:

    1. 'oflm-build' (Conda Env)
       - Path: C:\Users\user\miniforge3\envs\oflm-build
       - Purpose: A one-time setup artifact. Used to run mlir-aie's 
         utils/iron_setup.py which creates the ironenv venv.
       - Contains: Python 3.13, pip, numpy, ml_dtypes, etc.

    2. 'ironenv' (Python venv) - REQUIRED BY THIS SCRIPT
       - Path: C:\Users\user\lyb-llm-host\projects\iron\mlir-aie\ironenv
       - Purpose: The actual toolchain used for kernel export (aiecc, peano, mlir_aie).
       - Contains: mlir_aie 1.4.2, llvm-aie 22.0.0.2026101901, plus mlir_aie deps.

    Additional System Dependencies:
    - XRT SDK at C:\Xilinx\XRT (added to PATH by iron_env.ps1 for aiebu-asm.exe, xclbinutil.exe)
    - LLVM via Scoop (llvm-objcopy.exe) on system PATH.

    ------------------------------------------------------------------
    FRESH MACHINE REBUILD SEQUENCE (If ironenv is missing)
    ------------------------------------------------------------------
    1. conda create -n oflm-build python=3.13 -y
    2. git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
       (into C:\Users\user\lyb-llm-host\projects\iron\mlir-aie)
    3. & "C:\Users\user\miniforge3\envs\oflm-build\python.exe" .\utils\iron_setup.py
       (This creates the ironenv venv)
    4. pip install --upgrade mlir_aie==1.4.2 (or latest compatible)
    5. Ensure Scoop LLVM is installed.
    ------------------------------------------------------------------
#>

# 0. Check if we are in the specific 'ironenv' venv
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonPath -or ($pythonPath -notmatch "ironenv\\Scripts\\python\.exe$")) {
    Write-Host "Error: You are not in the 'ironenv' environment." -ForegroundColor Red
    Write-Host "Active Python: $pythonPath" -ForegroundColor Gray
    Write-Host "Expected:      C:\Users\user\lyb-llm-host\projects\iron\mlir-aie\ironenv\Scripts\python.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To enter the environment, dot-source the activation script:" -ForegroundColor Yellow
    Write-Host "    . C:\Users\user\lyb-llm-host\projects\iron\mlir-aie\iron_env.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Once your prompt looks like '(ironenv) PS C:\...>', run this script again." -ForegroundColor Yellow
    exit 1
}

# 1. Stop any running oflm process
Get-Process oflm -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Setup MLIR-AIE environment (ensures XRT and ironenv paths are correct)
cd C:\Users\user\lyb-llm-host\projects\iron\mlir-aie
. .\iron_env.ps1

# 3. Navigate to the project directory
cd C:\Users\user\lyb-llm-host\projects\OpenFlowLM-Next

# 4. Delete your entire open_kernels/designs/dense/build_* directory
Remove-Item -Path .\open_kernels\designs\dense\build_* -Recurse -Force -ErrorAction SilentlyContinue

# 5. Export the kernels
python .\open_kernels\export_qwen36_kernels.py --model-dir "$env:USERPROFILE\.flm\models\Ornith-1.5-9B-NPU2" --force

# 6. Verify manifest settings
$m = Get-Content .\src\xclbins\Ornith-1.5-9B-NPU2\open_kernels\manifest.json -Raw | ConvertFrom-Json
"kv_row = $($m.layout.kv_row), kv_quant = $($m.layout.kv_quant)"   # expect 1088, q8

# 7. Build the project
cmake --build --preset windows-vs18 --target oflm

# 8. Launch the server
$env:OFLM_DUMP_KV="0" # Debug
# .\build-vs18\src\oflm.exe serve nanbeige4.1:3b --host 10.0.10.10 --port 52632 --ctx-len 4096 --pmode turbo --quiet
# .\build-vs18\src\oflm.exe serve ornith1.5:9b --host 10.0.10.10 --port 52632 --ctx-len 4096 --pmode turbo --quiet
.\build-vs18\src\oflm.exe serve ornith15moe:35b --host 10.0.10.10 --port 52632 --ctx-len 4096 --pmode turbo --quiet