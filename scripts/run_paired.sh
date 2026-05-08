#!/bin/bash
# K-967 paired ON/OFF + torch reference benchmark, separate-process per arm
# Usage: ./run_paired.sh M N K dtype rounds
set -euo pipefail
M=$1; N=$2; K=$3; DT=$4; ROUNDS=${5:-30}
OUT_DIR=/home/ryaswann/mc2-workspaces/K-967/output
mkdir -p $OUT_DIR
CSV=$OUT_DIR/paired_${M}x${N}x${K}_${DT}.csv
> $CSV

echo "M,N,K,dtype,backend,TB_K967_PROTO,iter,time_ms" > $CSV.header
mv $CSV.header $CSV
echo "M,N,K,dtype,backend,TB_K967_PROTO,iter,time_ms" > $CSV

cd /home/ryaswann/mc2-workspaces/K-967/repos/tritonblas

# OFF
TB_K967_PROTO=0 HIP_VISIBLE_DEVICES=0 python3 /home/ryaswann/mc2-workspaces/K-967/scripts/bench_one.py $M $N $K $DT tb 10 20 $ROUNDS $CSV
# ON
TB_K967_PROTO=1 HIP_VISIBLE_DEVICES=0 python3 /home/ryaswann/mc2-workspaces/K-967/scripts/bench_one.py $M $N $K $DT tb 10 20 $ROUNDS $CSV
# torch reference
HIP_VISIBLE_DEVICES=0 python3 /home/ryaswann/mc2-workspaces/K-967/scripts/bench_one.py $M $N $K $DT torch 10 20 $ROUNDS $CSV
