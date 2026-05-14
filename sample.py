# built-in libs
from collections import defaultdict
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

# deps
from configs import dit_imagenet, dit_imagenet_repa
from data import local_imagenet_dataset, utils as data_utils
from eval import fid
from interfaces import continuous
from utils import (
    checkpoint as ckpt_utils,
    initialize as init_utils,
    logging_utils,
    sharding_utils,
    wandb_utils,
    visualize as vis_utils,
)


if __name__ == "__main__":
    repa = False

    if repa:
        config = dit_imagenet_repa.get_config('imagenet_256-XL_2')
        workdir = "gs://will-data/jmt/024_Mar-31-REPA-XL-256"
    else:
        workdir = "gs://will-data/jmt/019_Mar-27-SiT-XL-256-0325-jit-fsdp"
        config = dit_imagenet.get_config('imagenet_256-XL_2')

    encoder, model, optimizer, sampler, ema, learning_rate_fn = init_utils.build_models(config)

    if repa:
        model = init_utils.instantiate_repa(
            config, model, feature_dim=768
        )  # <-- will return a repa wrapper
        optimizer, _ = init_utils.instantiate_optimizer(config, model)

    ckpt_mngr = ckpt_utils.build_checkpoint_manager(
        workdir, **config.checkpoint.options
    )
    restore_step = ckpt_mngr.latest_step()

    opt_graph, opt_rng_state, opt_state = nnx.split(optimizer, nnx.RngKey, ...)
    _, _, ema_state = nnx.split(ema, nnx.RngKey, ...)
    loaded_state, loaded_rng_state, loaded_ema_state = ckpt_utils.restore_checkpoints(
        workdir, restore_step, opt_state, opt_rng_state, ema_state, mngr=ckpt_mngr
    )
    print('Restoring done.')
    optimizer = nnx.merge(opt_graph, opt_rng_state, loaded_state)

    rng = jax.random.PRNGKey(1)
    n = jax.random.normal(rng, (16, 32, 32, 4))
    y = jax.random.randint(rng, (16,), 0, 1000, dtype=jnp.uint8)

    print('Sampling start.')
    t = jnp.concatenate(
        [jnp.linspace(1., 0.7, 28), jnp.linspace(0.5, 0., 4)], axis=0
    )
    x = sampler.sample(rng, optimizer.model, n, y=y, guidance_scale=2.0, custom_timegrid=t[:28])
    img = wandb_utils.array2grid(encoder.decode(x))
    print('Sampling done.')

    PIL.Image.fromarray(img).save('repa_grid.png' if repa else 'grid.png')
