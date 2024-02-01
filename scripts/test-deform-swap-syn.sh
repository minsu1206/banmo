# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Use precomputed root body poses
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
deform_path=$6
num_epochs=60
batch_size=256

model_prefix=test-deform-swap-syn-$seqname

# mode: line load
# difference from template.sh
# 1) remove pose_net path flag
# 2) add use_rtk_file flag
savename=${model_prefix}
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --use_rtk_file \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --freeze_root \
  --freeze_proj \
  --proj_end 1 \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --deform_path $deform_path \
  --reinit_bone_steps 1 \
  --warmup_steps 1
