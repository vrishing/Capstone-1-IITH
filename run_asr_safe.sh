#!/usr/bin/env bash
set -u
MIN_FREE_MIB=6000

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
read -r GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                     | tr -d ' ' | sort -t, -k2 -n -r | head -1 | tr ',' ' ')
echo "best GPU: ${GPU:-none} with ${FREE:-0} MiB free"
if [ "${FREE:-0}" -lt "$MIN_FREE_MIB" ]; then
  echo "No GPU has $MIN_FREE_MIB MiB free. Not starting."; exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { nice -n 19 python run_pipeline.py "$@" 2>&1 | tee -a data/asr.log; }

run --playlist mahabharata --sample 10 --limit 10 --through asr_done
run --playlist nptelml --through asr_done
run --playlist mahabharata --through asr_done
