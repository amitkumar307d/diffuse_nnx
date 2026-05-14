"""File containing the main entry point for training & evaluation."""

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

    bucket = FLAGS.bucket
    prefix = FLAGS.prefix
    workdir = FLAGS.workdir

    if jax.process_count() > 1:
        import jax._src.distributed as jax_distributed
        if int(os.environ.get("JAX_PROCESS_INDEX", 0)) == 0:
            logging.info('Current commit: ')
            os.system('git show -s --format=%h')
            logging.info('Current dir: ')
            os.system('pwd')
            
            if gcloud_utils.directory_exists(bucket, prefix, workdir):
                index = gcloud_utils.get_directory_index(bucket, prefix, workdir)
                workdir = f"gs://{bucket}/{prefix}/{index:03d}_{workdir}"
            else:
                workdir = create_experiment_dir(bucket, prefix, workdir)
            jax_distributed.global_state.client.key_value_set("gcs_workdir", workdir)
        else:
            workdir = jax_distributed.global_state.client.blocking_key_value_get("gcs_workdir", 300000)
    else:
        if int(os.environ.get("JAX_PROCESS_INDEX", 0)) == 0:
            logging.info('Current commit: ')
            os.system('git show -s --format=%h')
            logging.info('Current dir: ')
            os.system('pwd')
        if gcloud_utils.directory_exists(bucket, prefix, workdir):
            index = gcloud_utils.get_directory_index(bucket, prefix, workdir)
            workdir = f"gs://{bucket}/{prefix}/{index:03d}_{workdir}"
        else:
            workdir = create_experiment_dir(bucket, prefix, workdir)     
    
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX process: %d / %d',
                 int(os.environ.get("JAX_PROCESS_INDEX", 0)), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f'process_index: {int(os.environ.get("JAX_PROCESS_INDEX", 0))}, '
        f'process_count: {jax.process_count()}'
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, workdir, 'workdir'
    )

    logging.info(FLAGS.config)

    if jax.local_devices()[0].platform != 'tpu':
        logging.error('Not using TPU. Exit.')
        exit()
    
    logging.info("Start training with trainer: %s", FLAGS.config.trainer)
    trainer = get_trainers(FLAGS.config.trainer)
    trainer.train_and_evaluate(FLAGS.config, workdir)


if __name__ == '__main__':
    from absl import flags
    import orbax.checkpoint
    flags.FLAGS.experimental_orbax_use_distributed_process_id = True
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import os
    print(">>> Reached jax.distributed.initialize()", flush=True)
    jax.distributed.initialize(coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"], num_processes=int(os.environ["JAX_PROCESS_COUNT"]), process_id=int(os.environ["JAX_PROCESS_INDEX"]))
    print(">>> Passed jax.distributed.initialize()", flush=True)

    if not (jax.process_index() == 0):  # not first process
        logging.set_verbosity(logging.ERROR)  # disable info/warning
    logging_utils.set_time_logging(logging)
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)