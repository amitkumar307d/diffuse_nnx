"""File containing evaluation code for calculating the FID score."""

# built-in libs
import functools
import math
import pickle
from typing import Any, Callable, Iterable

# external libs
from absl import logging
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections
import numpy as np
import torch
from tqdm import tqdm

# deps
from data import utils as data_utils
from eval import utils
from samplers import samplers
from utils import wandb_utils, sharding_utils


def calculate_stats_for_iterable(
    image_iter: Iterable[jnp.ndarray] | jnp.ndarray,
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,
    batch_size: int = 64,
    num_eval_images: int | None = None,
    verbose: bool = False,
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, np.ndarray]:
    """Calculate the statistics (mean and covariance) for an iterable/loader of images.

    Leverages modern JAX global device sharding to distribute the image batch over
    the data axis of the device mesh, and utilizes host process all-gathers to 
    accurately compute and aggregate overall metrics globally across all processes.

    Args:
        image_iter: An iterable (e.g. PyTorch DataLoader) or a raw NumPy/JAX Array of images.
        detector: JIT-compiled InceptionV3 model forward function.
        detector_params: Replicated parameters for the InceptionV3 detector.
        batch_size: Local batch size per host VM.
        num_eval_images: Optional maximum number of images to evaluate.
        verbose: If True, displays a progress bar.
        mesh: Global JAX device mesh for multi-host shard routing.

    Returns:
        A dictionary containing 'mu' (mean) and 'sigma' (covariance) of features.
    """
    local_batch_size = batch_size
    global_batch_size = batch_size * jax.process_count()
    
    if isinstance(image_iter, np.ndarray) or isinstance(image_iter, jnp.ndarray):
        assert len(image_iter.shape) == 4, 'Image array should have shape (N, H, W, C)'
        image_iter = image_iter.reshape(-1, global_batch_size, *image_iter.shape[1:])
        process_fn = lambda x: x
    else:
        process_fn = lambda x: x[0].permute([0, 2, 3, 1]).numpy()

    total_num_images = 0

     # TODO: remove the hardcoding here
    running_mu = np.zeros(2048, dtype=np.float64)
    running_cov = np.zeros((2048, 2048), dtype=np.float64)

    for i, batch in enumerate(
        tqdm(image_iter, desc='Calculating statistics', disable=not verbose)
    ):
        batch = process_fn(batch)
        batch = sharding_utils.make_fsarray_from_local_slice(batch, mesh.devices.flatten())
        
        batch_features = detector(detector_params, batch)
        total_num_images += batch_features.shape[0]

        batch_features = sharding_utils.get_local_slice_from_fsarray(batch_features)
        batch_features = np.asarray(batch_features, dtype=np.float64)

        if num_eval_images is not None and total_num_images > num_eval_images:
            batch_features = batch_features[:(num_eval_images - total_num_images)]

        running_mu += np.sum(batch_features, axis=0)
        running_cov += np.matmul(batch_features.T, batch_features)

        if num_eval_images is not None and total_num_images >= num_eval_images:
            total_num_images = num_eval_images
            break
            
    # Globally aggregate across all hosts
    running_mu = np.sum(jax.experimental.multihost_utils.process_allgather(running_mu), axis=0)
    running_cov = np.sum(jax.experimental.multihost_utils.process_allgather(running_cov), axis=0)
    
    mu = running_mu / total_num_images
    cov = (running_cov - np.outer(mu, mu) * total_num_images) / (total_num_images - 1)
    
    return {'mu': mu, 'sigma': cov}


def calculate_real_stats(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.Dataset,
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,
    verbose: bool = False,
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, np.ndarray]:
    """Calculate the statistics for real images using modern JIT.
    
    If config.data.stat_dir is provided, loads pre-computed statistics directly from disk/GCS.
    Otherwise, streams the real dataset via a distributed loader and extracts the stats.

    Args:
        config: Configuration dict containing directory and evaluation settings.
        dataset: PyTorch Dataset containing the real ImageNet images.
        detector: JIT-compiled InceptionV3 model forward function.
        detector_params: Replicated parameters for the InceptionV3 detector.
        verbose: If True, shows a progress bar.
        mesh: Global JAX device mesh for multi-host shard routing.

    Returns:
        A dictionary containing the mu (mean) and sigma (covariance) feature statistics.
    """
    
    if config.data.get('stat_dir'):
        logging.info(f'Loading stats from {config.data.stat_dir}...')
        with open(config.data.stat_dir, 'rb') as f:
            stats = pickle.load(f)
        logging.info(f'Loaded stats from {config.data.stat_dir}...')
        return {'mu': stats['fid']['mu'], 'sigma': stats['fid']['sigma']}

    # build distributed loader
    loader, _ = utils.build_eval_loader(
        dataset, config.eval.inception_batch_size * jax.local_device_count(), config.data.num_workers
    )
    return calculate_stats_for_iterable(loader, detector, detector_params, verbose=verbose, mesh=mesh)


def calculate_cls_fake_stats(
    config: ml_collections.ConfigDict,
    rngs: nnx.Rngs,

    # generator
    sampler: samplers.Samplers,
    generator: nnx.Module,

    # encoder
    encoder: nnx.Module,

    # detector
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,

    # guidance
    guide_generator: nnx.Module | None = None,
    guidance_scale: float = 1.0,
    all_eval_sample_nums: list[int] = [50000],

    # utils
    save_samples_path: str | None = None,
    mesh: Mesh | None = None
)-> dict[str, np.ndarray]:
    """Extract and calculate the statistics for class-conditioned synthesized images.
    
    Args:
    - config: Overall config for experiment.
    - rng: nnx Rngs stream for random number generation.

    - generator: Generator.
    - generator_params: Parameters for the generator.

    - encoder: Encoder.

    - detector: Function to extract features. **Note: detector is assumed to be pmap / pjit'd.**
    - detector_params: Parameters for the detector. **Note: detector_params is assumed to be processed to match detector.**

    - guide_generator: Guiding generator.
    - guide_generator_params: Parameters for the guiding generator.
    - guidance_scale: scale for generation guidance.
    - all_eval_sample_nums: a list of number of total samples to generate.

    Returns:
    - stats: Inception statistics for the images.
    """

    batch_size = config.eval.batch_size * jax.local_device_count()
    sample_size = config.data.image_size // config.encoder.get('downsample_factor', 1)
    sample_channels = config.network.in_channels

    if guide_generator is None:
        guide_generator = generator

    @functools.partial(
        jax.jit,
        static_argnums=(5, 6, 7),
    )
    def sample_step(gen_state, g_gen_state, rngs_state, x, c,
                    gen_graph, g_gen_graph, rngs_graph):
        gen = nnx.merge(gen_graph, gen_state)
        g_gen = nnx.merge(g_gen_graph, g_gen_state)
        rng = nnx.merge(rngs_graph, rngs_state)
        return sampler.sample(
            rng, gen, x, y=c,
            g_net=g_gen, guidance_scale=guidance_scale
        )

    max_eval_samples = max(all_eval_sample_nums)
    eval_iters = math.ceil(
        max_eval_samples / (batch_size * jax.process_count())
    )

    # gather state to reduce sampling overhead
    repl_sharding = jax.sharding.NamedSharding(mesh, P())
    def sync_state(state: nnx.State):
        return state
    p_sync_state = jax.jit(sync_state, out_shardings=repl_sharding)
    gen_graph, gen_state = nnx.split(generator)
    gen_state = p_sync_state(gen_state)

    g_gen_graph, g_gen_state = nnx.split(guide_generator)
    g_gen_state = p_sync_state(g_gen_state)

    total_num_samples = 0
    per_process_samples = []

    for i in range(eval_iters):
        x = jax.random.normal(
            rngs(), (batch_size, sample_size, sample_size, sample_channels)
        )
        c = jax.random.randint(
            rngs(), (batch_size,), 0, config.network.num_classes
        )
        x = sharding_utils.make_fsarray_from_local_slice(x, mesh.devices.flatten())
        c = sharding_utils.make_fsarray_from_local_slice(c, mesh.devices.flatten())
        rngs_graph, rngs_state = nnx.split(rngs)
        samples = sample_step(
            gen_state, g_gen_state, rngs_state, x, c,
            gen_graph, g_gen_graph, rngs_graph,
        )
        # ensure no additional batch axis present
        assert samples.ndim == 4, 'Samples should have shape (N, H, W, C)'

        per_process_samples.append(
            # np.asarray(jax.device_get(samples))
            sharding_utils.get_local_slice_from_fsarray(encoder.decode(samples))
        )
        total_num_samples = total_num_samples + samples.shape[0]

        logging.info(f'Generated {total_num_samples} samples')

    per_process_samples = np.concatenate(per_process_samples, axis=0)

    all_stats = {}
    for num_eval_sampels in all_eval_sample_nums:
        all_stats[num_eval_sampels] = calculate_stats_for_iterable(
            per_process_samples, detector, detector_params, config.eval.inception_batch_size, num_eval_sampels, mesh=mesh
        )
    
    if save_samples_path is not None:
        # directly save to gcs
        pass

    # TODO: check if this is necessary
    utils.lock()
    
    return all_stats

    
def calculate_fid(
    config: ml_collections.ConfigDict,

    # real dataset
    dataset: torch.utils.data.Dataset,

    # generator
    sampler: samplers.Samplers,
    generator: nn.Module,
    encoder: nn.Module,

    # sampler
    guidance_scale: float = 1.0,

    # guidance
    guide_generator: nn.Module | None = None,

    # utils
    sample_sizes: list[int] = [10000],
    step: int = 0,
    mesh: Mesh | None = None
) -> dict[str, float]:
    """Calculate the FID score betwee the synthesized images and real dataset."""
    logging.info(
        f"Calculating FID-{max(sample_sizes) // 1000}K scores with cfg={guidance_scale}..."
    )
    # fix rngs for each evaluation
    rngs = nnx.Rngs(config.eval.seed + jax.process_index())

    detector_params, detector = utils.get_detector(config, mesh)

    real_stats = calculate_real_stats(config, dataset, detector, detector_params, mesh=mesh)
    fake_stats = calculate_cls_fake_stats(
        config, rngs, sampler, generator, encoder, detector, detector_params,
        guide_generator, guidance_scale, sample_sizes, config.eval.save_samples_path, mesh=mesh
    )


    all_fid_scores = {}
    for num_samples, stats in fake_stats.items():
        fid = utils.calculate_fid(stats, ref_stats=real_stats)
        all_fid_scores[num_samples] = fid
        fid_label = f"FID-{num_samples // 1000}K (cfg={guidance_scale})"
        wandb_utils.log({fid_label: fid, 'train_step': step})
        logging.info(f"{fid_label}: {fid}")

    return all_fid_scores
