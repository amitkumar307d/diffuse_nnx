"""File containing utilities for generating visualization."""

# built-in libs
import functools

# external libs
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections

# deps
from utils import wandb_utils, sharding_utils
from samplers import samplers


@functools.partial(
    jax.jit,
    static_argnums=(5, 6, 7, 8, 9),
)
def _sample_jit(net_state, g_net_state, rngs_state, n, c,
                net_graph, g_net_graph, rngs_graph, guidance_scale, sampler):
    net = nnx.merge(net_graph, net_state)
    g_net = nnx.merge(g_net_graph, g_net_state)
    rngs = nnx.merge(rngs_graph, rngs_state)
    return sampler.sample(
        rngs, net, n, y=c, g_net=g_net,
        guidance_scale=guidance_scale
    )


def sample_fn(net, g_net, encoder, rngs, n, c, guidance_scale, sampler):
    net_graph, net_state = nnx.split(net)
    g_net_graph, g_net_state = nnx.split(g_net)
    rngs_graph, rngs_state = nnx.split(rngs)
    x = _sample_jit(
        net_state, g_net_state, rngs_state, n, c,
        net_graph, g_net_graph, rngs_graph, guidance_scale, sampler,
    )
    return encoder.decode(x)


def visualize(
    config: ml_collections.ConfigDict,
    net: nnx.Module,
    ema_net: nnx.Module,
    encoder: nnx.Module,
    sampler: samplers.Samplers,
    step: int, 
    g_net: nnx.Module | None = None,
    guidance_scale: float | None = None,
    mesh: Mesh | None = None,
) -> jnp.ndarray:
    """Generate and log samples from the model.
    
    Args:
    - config: configuration for the training.
    - net: nnx.Module, the network for training.
    - ema_net: nnx.Module, the ema network.
    - encoder: nnx.Module, the encoder for training.
    - n: jnp.ndarray, the initial noise.
    - c: jnp.ndarray, the initial condition.
    - guidance_scale: float, the guidance weight for the guidance network.
    - sampler: samplers.Samplers, the sampler for the network.

    Returns:
    - x: jnp.ndarray, the generated sample.
    """

    image_size = config.data.image_size // config.encoder.get('downsample_factor', 1)
    num_samples = config.visualize.num_samples // jax.process_count()

    # fix rngs for each visualization
    rngs = nnx.Rngs(config.eval.seed + jax.process_index())

    # generate the noise
    n = jax.random.normal(rngs(), (num_samples, image_size, image_size, config.network.in_channels))
    c = jax.random.randint(rngs(), (num_samples, ), 0, config.network.num_classes)

    n = sharding_utils.make_fsarray_from_local_slice(n, mesh.devices.flatten())
    c = sharding_utils.make_fsarray_from_local_slice(c, mesh.devices.flatten())

    logging.info("Generating model samples...")
    
    net.eval()
    x = sample_fn(
        net, g_net if g_net is not None else net,
        encoder, rngs, n, c, 1.0, sampler
    )
    x = jax.experimental.multihost_utils.process_allgather(x, tiled=True)
    wandb_utils.log_images(x, 'network', step=step)
    net.train()

    logging.info("Generating EMA samples...")
    x = sample_fn(
        ema_net, g_net if g_net is not None else ema_net,
        encoder, rngs, n, c, 1.0, sampler
    )
    x = jax.experimental.multihost_utils.process_allgather(x, tiled=True)
    wandb_utils.log_images(x, 'ema_network', step=step)

    if config.visualize.guidance_scale > 1.0 or guidance_scale is not None and guidance_scale > 1.0:
        logging.info("Generating EMA samples with guidance...")

        guidance_scale = config.visualize.guidance_scale if guidance_scale is None else guidance_scale

        x = sample_fn(
            ema_net, g_net if g_net is not None else ema_net,
            encoder, rngs, n, c, guidance_scale, sampler
        )
        x = jax.experimental.multihost_utils.process_allgather(x, tiled=True)
        wandb_utils.log_images(x, f'ema_network_cfg={guidance_scale}', step=step)


def visualize_reconstruction(
    config: ml_collections.ConfigDict,
    encoder: nnx.Module,
    x: jnp.ndarray,
    mesh: Mesh | None = None,
) -> jnp.ndarray:
    """Plotting the image"""
    # x = sharding_utils.make_fsarray_from_local_slice(x, mesh.devices.flatten())
    wandb_utils.log_images(
        (x.astype(jnp.float32) * 127.5 + 128).clip(0, 255).astype(jnp.uint8),
        'encoder_original',
        step=0
    )

    logging.info(f"Reconstructing images...")

    z = encoder.encode(x, key=jax.random.PRNGKey(0))  # <-- okay to fix key
    x_rec = encoder.decode(z)

    wandb_utils.log_images(x_rec, 'encoder_reconstructed', step=0)