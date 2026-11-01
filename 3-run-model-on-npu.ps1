# 1. Stop any running oflm process
Get-Process oflm -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Setup MLIR-AIE environment (ensures XRT and ironenv paths are correct)
# cd C:/Users/user/lyb-llm-host/projects/iron/mlir-aie
# . ./iron_env.ps1
# Ensure open kernel is built using something like this:
# (ironenv) PS C:\Users\user\lyb-llm-host\projects\OpenFlowLM-Next> python ./open_kernels/export_qwen36_kernels.py `
# >>     --model-dir "$env:USERPROFILE/.oflm/models/Qwen3.8-Distilled-9B-NPU2" `
# >>     --force

# 3. Setup Environment
# Remove-Item Env:/OFLM_MODEL_PATH
$env:OFLM_MODEL_PATH = "C:/Users/user/.oflm"
$env:OFLM_KEEP_FAILED = 1
$env:OFLM_OPEN_DISPATCH_LOG = 1
$env:OFLM_OPEN_GEMM_BLOCK = 0
$env:OFLM_OPEN_ROUTE_SENTINEL = 1
$env:OFLM_OPEN_STEP_TRACE = 1
$env:Path = "C:/Users/user/miniforge3/envs/oflm-build/Library/bin;C:/Users/user/miniforge3/Library/bin;" + $env:Path
# Deduplicate the PATH for the current session
$env:Path = ($env:Path -split ';' | Select-Object -Unique) -join ';'

# 4. Navigate to the project directory
cd C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next

# 5. CHOOSE MODEL AND RUN

# ornith1.5:9b
# ./build-vs18/src/oflm.exe serve ornith1.5:9b --host 10.0.10.10 --port 52625 --ctx-len 262144 --pmode turbo --quiet

# qwen3.8:9b
./build-vs18/src/oflm.exe serve qwen3.8:9b --host 10.0.10.10 --port 52625 --ctx-len 262144 --pmode turbo

# groot:19b
# ./build-vs18/src/oflm.exe serve groot:19b --host 10.0.10.10 --port 52625 --ctx-len 32768 --pmode turbo --prefill-mode fast
