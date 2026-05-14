# run remote

CONFIG="rae_imagenet"
BATCH_SIZE=1024
WORKDIR="Nov-19-RAE-XL-512"
BUCKET="$JMT_GCS_BUCKET"

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export WANDB_ENTITY="nm3607"

source /mnt/disks/imagenet/ENTER/bin/activate
conda activate will_env

WANDB_API_KEY="3c57a0b61e31ecc5db2d791aa2dce4637d94cc7c" python main.py \
    --workdir=$WORKDIR \
    --bucket="willis-storage" \
    --config=configs/$CONFIG.py:imagenet_raw_512-XL_1 \
    --config.data.batch_size=$BATCH_SIZE \
    --config.standalone_eval=False \
    --config.project_name='jmt' \
    --config.eval.on_load=False \
    --config.visualize.on=False \
    --config.interface.train_time_dist_type=logitnormal \
    --config.exp_name='RAE-XL-512-1119'
    # --config.pretrained_ckpt="gs://$BUCKET/jmt/pretrained_ckpts/nnx/DDTXL_256" \
