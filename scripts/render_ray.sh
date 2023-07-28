seqname=$1
model_path=$2
test_frames=$3

testdir=${model_path%/*} # %: from end
add_args=${*: 3:$#-1}
prefix=$testdir/$seqname-$test_frames

echo $testdir
echo $model_path
echo "visualize_ray.py"
echo $add_args
python visualize_ray.py --flagfile=$testdir/opts.log \
                  --seqname $seqname \
                  --model_path $model_path \
                  --test_frames $test_frames \
                  $add_args