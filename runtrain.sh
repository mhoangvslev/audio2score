if [[ $# -lt 3 ]] ; then
    echo 'missing arguments: ./runtrain.sh config manifest_id model_id [args]'
    exit -1
fi

config=$1
manid=$2
modid=$3
shift 3

cuda=""
if [[ $1 == "--cuda" ]]; then
    cuda=$1
    shift;
fi

# Or python -m multiproc for multi-GPU training
if [[ $cuda == "--cuda" ]]; then
    echo "Model will be trained using CUDA"
    python train.py --cuda --config-path ${config} --train-manifest train_${manid}.csv --val-manifest val_${manid}.csv --labels-path labels_${manid}.json --num-workers 4 --model-path models/${modid}.pth $*
else
    echo "Model will be trained using CPU"
    python train.py --config-path ${config} --train-manifest train_${manid}.csv --val-manifest val_${manid}.csv --labels-path labels_${manid}.json --num-workers 4 --model-path models/${modid}.pth $*
fi