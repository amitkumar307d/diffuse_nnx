"""File containing the util functions for evaluation."""

# built-in libs
import math
import os
import requests
import tempfile
import functools

# external libs
from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import scipy
import torch
from tqdm import tqdm

# deps
from eval import inception
from samplers import samplers


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]


def download(url, ckpt_dir=None):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'jax_fid')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


def all_gather(x: jnp.ndarray) -> jnp.ndarray:
    """convenient wrapper for jax.lax.all_gather"""
    assert x.shape[0] == jax.local_device_count(), f"Expected first dimension to be the number of local devices, got {x.shape[0]} != {jax.local_device_count()}"
    all_gather_fn = lambda x: jax.lax.all_gather(x, axis_name='data', tiled=True)
    all_gathered = jax.pmap(all_gather_fn, axis_name='data')(x)[0]
    return all_gathered


def lock():
    """Hold the lock until all processes sync up."""
    all_gather(jnp.ones((jax.local_device_count(),1))).block_until_ready()


def build_keep_indices(
    item_subset: list[int],
    batch_size: int,
    len_dataset: int
):
    """
    This function simulates the behavior of a DataLoader with the item_subset sampler.
    The intent is to find and remove the indices of images that are processed twice to avoid 
    biasing FID.
    """
    keep_indices = jnp.array(item_subset) < len_dataset
    final_indices = []
    for batch_start in range(0, keep_indices.shape[0], batch_size):
        indices_in = keep_indices[batch_start:batch_start+batch_size].reshape((jax.local_device_count(), -1))
        batch_keep_indices = all_gather(indices_in).reshape(-1)
        final_indices.append(batch_keep_indices)
    return final_indices


def build_eval_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 8,
) -> torch.utils.data.DataLoader:
    """Build the dataloader for evaluation."""
    dataset_len = len(dataset)
    n = jax.process_count()
    pad_factor = batch_size
    
    # pad the dataset to be divisible by the batch size and local_device_count:
    if (pad_factor // n) % jax.local_device_count() != 0:
        pad_factor *= jax.local_device_count()

    dataset_len = int(math.ceil(dataset_len / pad_factor)) * pad_factor

    item_subset = [(i * n + jax.process_index()) for i in range((dataset_len - 1) // n + 1)]
    keep_indices = build_keep_indices(item_subset, batch_size, len(dataset))
    item_subset = [i % len(dataset) for i in item_subset]
    loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=item_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,  # important
        worker_init_fn=None,
        persistent_workers=True,
        timeout=60.0
    )
    return loader, keep_indices


def get_detector(config: ml_collections.ConfigDict, mesh: jax.sharding.Mesh):
    """Get the sampler for fid evaluation."""
    if config.eval.detector == 'inception':
        logging.info('Loading InceptionV3 model for FID calculation...')
        detector = inception.InceptionV3(pretrained=True)

        params = detector.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
        
        # Fully replicate parameters across the global mesh
        repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        params = jax.device_put(params, repl_sharding)

        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))

        @functools.partial(
            jax.jit,
            in_shardings=(repl_sharding, data_sharding),
            out_shardings=data_sharding,
        )
        def forward(params, x):
            x = x.astype(jnp.float32) / 127.5 - 1
            x = jax.image.resize(x, (x.shape[0], 299, 299, x.shape[-1]), method='bilinear')
            features = detector.apply(params, x, train=False).squeeze(axis=(1, 2))
            return features

        logging.info('InceptionV3 model loaded.')
        return params, forward
    else:
        # TODO: add DINOv2
        raise NotImplementedError


def calculate_fid(
    stats: dict[str, np.ndarray],
    ref_stats: dict[str, np.ndarray]
) -> float:
    """Calculate the FID score between stats and ref_stats."""

    m = np.square(stats['mu'] - ref_stats['mu']).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(stats['sigma'], ref_stats['sigma']), disp=False)
    return float(np.real(m + np.trace(stats['sigma'] + ref_stats['sigma'] - s * 2)))
