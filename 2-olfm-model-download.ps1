# 1. Setup Environment
$env:OFLM_MODEL_PATH = "C:/Users/user/.oflm"
$env:OFLM_KEEP_FAILED = 1
$env:OFLM_OPEN_DISPATCH_LOG = 1
$env:OFLM_OPEN_GEMM_BLOCK = 0
$env:OFLM_OPEN_ROUTE_SENTINEL = 1
$env:OFLM_OPEN_STEP_TRACE = 1
$env:Path = "C:/Users/user/miniforge3/envs/oflm-build/Library/bin;C:/Users/user/miniforge3/Library/bin;" + $env:Path
# Deduplicate the PATH for the current session
$env:Path = ($env:Path -split ';' | Select-Object -Unique) -join ';'

# 2. Setup MLIR-AIE environment (ensures XRT and ironenv paths are correct)
cd C:/Users/user/lyb-llm-host/projects/iron/mlir-aie
. ./iron_env.ps1

# 3. Navigate to the project directory
cd C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next

# 4. CHOOSE MODEL AND PACK

# ornith1.5:35b-a3b
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/Ornith-1.5-35B-A3B-NPU2 --tag ornith1.5:35b-a3b --family qwen3.6-moe --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# qwen3.8:9b (GGUF-not working yet)
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/Qwen3.8-Distilled-9B-NPU2-GGUF --tag qwen3.8:9b --family qwen35 --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# qwen3.8:9b
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/Qwen3.8-Distilled-9B-NPU2 --tag qwen3.8:9b --family qwen35 --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# lyb.lfm2:2.6b
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/LFM2-2.6B-NPU2 --tag lyb.lfm2:2.6b --family lfm2 --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# TODO:nB
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/Qwen3.8-Distilled-9B-NPU2 --tag qwen3.8:9b --family qwen35 --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# TODO:nB
# python ./utilities/oflm-add/oflm-add.py Atomic-Germ/Qwen3.8-Distilled-9B-NPU2 --tag qwen3.8:9b --family qwen35 --system-list "C:/Users/user/lyb-llm-host/projects/OpenFlowLM-Next/src/model_list.json"

# unsloth/Qwen3-4B-Thinking-2507-GGUF
./build-vs18/src/oflm.exe pack -i "C:/Users/user/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Thinking-2507-GGUF/snapshots/f40adb104d4d44aee52f398b60597c5866a973a3/Qwen3-4B-Thinking-2507-Q8_0.gguf" -o "C:/Users/user/.oflm/models/Qwen3-4B-Thinking-2507-NPU2" -s Qwen/Qwen3-4B-Thinking-2507
