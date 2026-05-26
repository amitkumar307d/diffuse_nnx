"""File containing the main entry point for training & evaluation."""

# import jax
# def spy(*args, **kwargs):
#     import traceback
#     traceback.print_stack()
#     return original_init(*args, **kwargs)

# # Hook into the backend initialization
# from jax._src import xla_bridge
# original_init = xla_bridge.get_backend
# xla_bridge.get_backend = spy

# built-in libs
import os
import warnings

# external libs
from absl import app, flags, logging
from clu import platform
import jax
from ml_collections import config_flags

# deps
from utils import logging_utils, gcloud_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('bucket', None, 'Google Cloud Storage bucket.')
flags.DEFINE_string('prefix', 'jmt', 'Prefix for the experiment directory.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True
)

def create_experiment_dir(bucket, prefix, workdir):
    """Create a new experiment directory in Google Cloud Storage."""
    num_files = gcloud_utils.count_directories(bucket, prefix)
    workdir = f"gs://{bucket}/{prefix}/{num_files:03d}_{workdir}"
    return workdir


def get_trainers(trainer):
    """Get the trainers for the experiment."""
    if trainer == 'DiT_ImageNet':
        from trainers import dit_imagenet
        return dit_imagenet
    else:
        raise ValueError(f'Unknown trainer: {trainer}')


def main(argv):
    """The main entry point."""
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    try:
        # Disable info/warning logs on non-leader processes immediately
        if jax.process_index() != 0:
            logging.set_verbosity(logging.ERROR)

        logging.info('JAX process: %d / %d',
                     jax.process_index(), jax.process_count())
        logging.info('JAX local devices: %r', jax.local_devices())

        bucket = FLAGS.bucket
        prefix = FLAGS.prefix
        workdir = FLAGS.workdir

        if gcloud_utils.directory_exists(bucket, prefix, workdir):
            index = gcloud_utils.get_directory_index(bucket, prefix, workdir)
            workdir = f"gs://{bucket}/{prefix}/{index:03d}_{workdir}"
        else:
            workdir = create_experiment_dir(bucket, prefix, workdir)
        
        if jax.process_index() == 0:
            logging.info('Current commit: ')
            os.system('git show -s --format=%h')
            logging.info('Current dir: ')
            os.system('pwd')
        
        platform.work_unit().set_task_status(
            f'process_index: {jax.process_index()}, '
            f'process_count: {jax.process_count()}'
        )
        platform.work_unit().create_artifact(
            platform.ArtifactType.DIRECTORY, workdir, 'workdir'
        )

        # Dynamically compute total_steps from epochs if specified (greater than 0)
        if FLAGS.config.get('epochs', 0) > 0:
            steps_per_epoch = FLAGS.config.data.num_train_samples / FLAGS.config.data.batch_size
            FLAGS.config.total_steps = int(steps_per_epoch * FLAGS.config.epochs)
            logging.info(f"Dynamically calculated total_steps from epochs: {FLAGS.config.total_steps} steps")

        logging.info(FLAGS.config)

        if jax.local_devices()[0].platform != 'tpu':
            logging.error('Not using TPU. Exit.')
            return  
        
        logging.info("Start training with trainer: %s", FLAGS.config.trainer)
        trainer = get_trainers(FLAGS.config.trainer)
        trainer.train_and_evaluate(FLAGS.config, workdir)

    finally:
        # Keep shutdown inside the finally block to release the ports on exit
        jax.distributed.shutdown()


if __name__ == '__main__':
    # 1. CRITICAL: Initialize distributed runtime BEFORE flag parsing touches configs
    jax.distributed.initialize()
    
    # 2. Configure logging utilities safely post-initialization
    logging_utils.set_time_logging(logging)
    
    # 3. Parse flags and run the main app logic
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)