
# built-in libs
from collections import defaultdict
import functools
import math
import os
import pickle
import time
import warnings

# external libs
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections
import numpy as np
import PIL
import torch
from torchvision import transforms
from tqdm import tqdm

# deps
from configs import dit_imagenet
from data import local_imagenet_dataset, utils as data_utils
from eval import fid, utils as eval_utils
from interfaces import continuous
from utils import (
    checkpoint as ckpt_utils,
    initialize as init_utils,
    logging_utils,
    sharding_utils,
    wandb_utils,
    visualize as vis_utils,
)

@functools.partial(
    nnx.jit,
    static_argnames=['sampler', 'guidance_scale']
)
def sample_fn(rngs, net, x, y, timegrid, sampler, guidance_scale):
    samples = sampler.sample(
        rngs, net, x, guidance_scale=guidance_scale, y=y, custom_timegrid=timegrid
    )
    return samples


if __name__ == "__main__":

    config = dit_imagenet.get_config('imagenet_256-XL_2')
    del config.data.stat_dir
    with config.unlocked():
        config.eval.inception_batch_size = 32
        config.sharding.strategy = [('.*', 'replicate')]

    encoder, model, optimizer, sampler, ema, learning_rate_fn = init_utils.build_models(config)

    if config.get('repa'):
        detector = init_utils.instantiate_detector(config)
        model = init_utils.instantiate_repa(
            config, model, feature_dim=detector.network.config.hidden_size
        )  # <-- will return a repa wrapper
        optimizer, _ = init_utils.instantiate_optimizer(config, model)

    workdir = 'gs://will-data/jmt/019_Mar-27-SiT-XL-256-0325-jit-fsdp'
    ckpt_mngr = ckpt_utils.build_checkpoint_manager(
        workdir, **config.checkpoint.options
    )
    step = ckpt_mngr.latest_step()
    opt_graph, opt_rng_state, opt_state = nnx.split(optimizer, nnx.RngKey, ...)
    _, _, ema_state = nnx.split(ema, nnx.RngKey, ...)
    loaded_state, loaded_rng_state, loaded_ema_state = ckpt_utils.restore_checkpoints(
        workdir, step, opt_state, opt_rng_state, ema_state, mngr=ckpt_mngr
    )

    mesh = sharding_utils.create_device_mesh(
        config.sharding.mesh,
        allow_split_physical_axes=config.sharding.get('mesh_allow_split_physical_axes', False)
    )

    (
        graphdef,
        state,
        ema_graphdef,
        ema_state,
        state_sharding,
        ema_state_sharding,
    ) = sharding_utils.update_model_sharding(
        opt_graph, loaded_state, loaded_rng_state, ema, loaded_ema_state,
        mesh=mesh, sharding_strategy=config.sharding.strategy
    )

    del opt_state, opt_rng_state, loaded_state, loaded_ema_state

    optimizer = nnx.merge(graphdef, state)
    ema = nnx.merge(ema_graphdef, ema_state)

    print("Restoring Done.")

    rngs = nnx.Rngs(42)

    # load val dataset
    path = '/home/nm3607/jmt/experimental'
    dataset = local_imagenet_dataset.ValDataset(
        root='/home/nm3607/imagenet/val',
        label_file='/home/nm3607/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt',
        transform=data_utils.build_transform(256),
    )

    detector_params, detector = eval_utils.get_detector(config)  # hardcoded to not scale for inception
    print('Inception Loaded.')
    if os.path.exists(os.path.join(path, 'val_stats.pkl')):
        with open(os.path.join(path, 'val_stats.pkl'), 'rb') as f:
            val_stats = pickle.load(f)
    else:
        val_stats = fid.calculate_real_stats(
            config,
            dataset,
            detector,
            detector_params,
            verbose=True,
        )

        with open(os.path.join(path, 'val_stats.pkl'), 'wb') as f:
            pickle.dump({'fid': val_stats}, f)
    
    ############ Sampling ############

    # vanilla FID sampling
    sample_batch_size = 32
    total_num_samples = 50000
    eval_iters = math.ceil(
        total_num_samples / (sample_batch_size * jax.process_count())
    )
    all_samples = []
    for i in tqdm(range(eval_iters), desc='Sampling'):
        x = jax.random.normal(
            rngs(), (sample_batch_size, 32, 32, 4)
        )
        y = jax.random.randint(
            rngs(), (sample_batch_size,), 0, config.network.num_classes
        )

        x = sharding_utils.make_fsarray_from_local_slice(x, mesh.devices.flatten())
        y = sharding_utils.make_fsarray_from_local_slice(y, mesh.devices.flatten())
        
        samples = sample_fn(
            rngs, ema.ema, x, y, jnp.linspace(1., 0., 33), sampler, 1.5
        )
        samples = encoder.decode(samples)
        samples = samples.astype(jnp.float32) / 127.5 - 1
        all_samples.append(jax.device_get(samples))
        del x, y, samples
    
    all_samples = np.concatenate(all_samples, axis=0)
    stats = fid.calculate_stats_for_iterable(
        all_samples, detector, detector_params, 1, total_num_samples, verbose=True
    )
    fid_res = eval_utils.calculate_fid(stats, val_stats['fid'])
    print(f"FID: {fid_res}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_batch_size,
        num_workers=8,
        shuffle=False,
    )

    # denoising fid sampling
    all_dsamples = []
    encoder.encoded_pixels = False
    for batch in tqdm(loader):
        images = batch[0].permute([0, 2, 3, 1]).numpy()
        y = batch[1].numpy()
        latents = encoder.encode(images)
        n = jax.random.normal(
            rngs(), (latents.shape[0], 32, 32, 4)
        )
        x = 0.5 * latents + 0.5 * n

        x = sharding_utils.make_fsarray_from_local_slice(x, mesh.devices.flatten())
        y = sharding_utils.make_fsarray_from_local_slice(y, mesh.devices.flatten())

        # samples
        samples = sample_fn(
            rngs, ema.ema, x, y, jnp.linspace(0.5, 0., 17), sampler, 1.5
        )
        samples = encoder.decode(samples)
        samples = samples.astype(jnp.float32) / 127.5 - 1
        all_dsamples.append(jax.device_get(samples))
        del x, y, samples
    all_dsamples = np.concatenate(all_dsamples, axis=0)
    stats = fid.calculate_stats_for_iterable(
        all_dsamples, detector, detector_params, 1, None, verbose=True
    )
    fid_res = eval_utils.calculate_fid(stats, val_stats['fid'])
    print(f"dFID: {fid_res}")
