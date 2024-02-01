#!/bin/bash
export OMP_NUM_THREADS=1
seqname=$1
gpus=$2
deform_path=$3 # additional
bash scripts/test-deform-swap-syn.sh $gpus $seqname 10001 "no" "no" $deform_path
# bash scripts/render_mgpu.sh $gpus $seqname logdir/test-deform-swap-syn-$seqname/params_latest.pth "0" 256