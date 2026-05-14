# run remote

CONFIG="dit_imagenet_repa"
BATCH_SIZE=256
WORKDIR="Jul-24-REPA-B-256-lyr-4"
BUCKET="$JMT_GCS_BUCKET"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching the training job.}"
: "${JMT_GCS_BUCKET:?Set JMT_GCS_BUCKET before launching the training job.}"

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592

WANDB_API_KEY="$WANDB_API_KEY" python main.py \
    --workdir=$WORKDIR \
    --bucket=$BUCKET \
    --config=configs/$CONFIG.py:imagenet_raw_256-B_2 \
    --config.data.batch_size=$BATCH_SIZE \
    --config.standalone_eval=False \
    --config.project_name='jmt' \
    --config.eval.on_load=False \
    --config.visualize.on=True \
    --config.interface.train_time_dist_type=uniform \
    --config.exp_name='REPA-B-256-0724-lyr-4' \
