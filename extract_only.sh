#!/bin/bash
export OMP_NUM_THREADS=1
seqname=$1
gpus=1
model_path=$2
testdir=${model_path%/*}
python extract.py --flagfile=$testdir/opts.log \
                    --seqname $seqname \
                    --model_path $model_path \
                    --test_frames "{0}" \
                    --sample_grid3d 256 --full_mesh --mc_threshold 0 \
                    --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2

# example
# bash extract_only.sh a-eagle logdir/known-cam-a-eagle-init/params_latest.pth
