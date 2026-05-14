"""File containing the PCA visualization of latent space / DiT features."""

# built-in libs

# external libs
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image

# deps
from configs import dit_imagenet
from data import local_imagenet_dataset
from utils import initialize


if __name__ == "__main__":


    config = dit_imagenet.get_config('imagenet_256-B_2')
    rng = jax.random.PRNGKey(0)
    n = np.asarray(jax.random.normal(rng, (1024, 768)))

    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir=config.data.data_dir,
        image_size=256,
        latent_dataset=True,
    )
    encoder = initialize.instantiate_encoder(config)

    latent = encoder.encode(np.transpose(dataset[2200][0], [1, 2, 0])[None, ...])
    image = np.asarray(encoder.decode(latent)[0])

    # Normalize the data
    n = StandardScaler().fit_transform(latent.reshape(-1, 4))
    pca = PCA(n_components=3)
    pca.fit(n)

    feature = pca.transform(n).reshape((32, 32, 3))
    feature = (feature * 127.5 + 128).clip(0, 255).astype(np.uint8)
    print(feature.shape)
    Image.fromarray(feature).convert('RGB').save('pca.png')
    Image.fromarray(image).convert('RGB').save('image.png')
    