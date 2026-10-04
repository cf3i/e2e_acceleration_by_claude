#!/bin/bash
# Background GPU monitor: gpumon.sh <out.csv>  (samples every 2s)
exec nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 2 -i "${CUDA_VISIBLE_DEVICES:-0}" > "$1"
