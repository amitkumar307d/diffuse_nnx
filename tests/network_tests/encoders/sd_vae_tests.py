"""File containing the unittest for the StabilityVAE encoder."""

# built-in libs
import os
import unittest

# external libs
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import PIL

# deps
from configs import dit_imagenet
from networks.encoders.sd_vae import StabilityVAE
from utils import initialize as init_utils


if __name__ == "__main__":
    
    config = dit_imagenet.get_config('imagenet_256-B_2')
    encoder = init_utils.instantiate_encoder(config)

    home_dir = os.path.expanduser("~")
    data = np.load(
        os.path.join(home_dir, 'jmt/tests/network_tests/encoders/test_latent.npy')
    )
    data = np.moveaxis(data, 0, -1)

    key = jax.random.PRNGKey(0)
    key, encode_key = jax.random.split(key)
    # data = jax.random.normal(encode_key, (256, 256, 3))
    res = encoder.decode(encoder.encode(data[None, ...], key=key, encoded_pixels=True))

    # print(res)

    PIL.Image.fromarray(np.asarray(res[0])).save('test.png')