# run remote

CONFIG="dit_imagenet"
BATCH_SIZE=256
WORKDIR="Apr-15-SiT-XL-pd-raw"
BUCKET="tpu-poc-test"
WANDB_API_KEY="3c57a0b61e31ecc5db2d791aa2dce4637d94cc7c"

# : "${WANDB_API_KEY:?Set WANDB_API_KEY before launching the training job.}"
# : "${JMT_GCS_BUCKET:?Set JMT_GCS_BUCKET before launching the training job.}"

echo $BUCKET

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592

WANDB_API_KEY="$WANDB_API_KEY" WANDB_ENTITY="nyu-visionx" python main.py \
    --workdir=$WORKDIR \
    --bucket=$BUCKET \
    --config=configs/$CONFIG.py:imagenet_raw_256-XL_2 \
    --config.data.batch_size=$BATCH_SIZE \
    --config.standalone_eval=False \
    --config.project_name='tpupoc' \
    --config.eval.on=True \
    --config.eval.on_load=False \
    --config.visualize.on=True \
    --config.interface.train_time_dist_type=uniform \
    --config.exp_name=${WORKDIR} \
