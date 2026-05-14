"""File containing the REPA wrapper for DiT."""

# built-in libs

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.transformers import dit_nnx

def build_mlp(hidden_size, projector_dim, feature_dim, rngs, dtype=jnp.float32):
    return nnx.Sequential(
        nnx.Linear(
            hidden_size, projector_dim, 
            dtype=dtype, precision=dit_nnx.PRECISION, rngs=rngs
        ),
        nnx.silu,
        nnx.Linear(
            projector_dim, projector_dim, 
            dtype=dtype, precision=dit_nnx.PRECISION, rngs=rngs
        ),
        nnx.silu,
        nnx.Linear(
            projector_dim, feature_dim,
            dtype=dtype, precision=dit_nnx.PRECISION, rngs=rngs
        ),
    )


class DiT_REPA(nnx.Module):
    
    def __init__(
        self,
        interface,
        *,
        feature_dim: int,
        repa_loss_weight: float,
        repa_depth: int,
        proj_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.interface = interface
        self.repa_depth = repa_depth
        self.repa_loss_weight = repa_loss_weight

        self.projector = build_mlp(
            interface.network.hidden_size, proj_dim, feature_dim,
            rngs=interface.network.rngs, dtype=dtype
        )

        self.interface.network.return_intermediate_features = True
    
    def loss(self, x: jnp.ndarray, x_feature: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        diffusion_loss, _, intermediate_features = self.interface(x, *args, return_aux=True, **kwargs)

        repa_feature = intermediate_features[self.repa_depth - 1]
        N, T, D = repa_feature.shape
        feature_proj = self.projector(repa_feature.reshape(-1, D)).reshape(N, T, -1)

        # TODO: update the following
        x_feature_norm = x_feature / jnp.linalg.norm(x_feature, axis=-1, keepdims=True)
        feature_proj_norm = feature_proj / jnp.linalg.norm(feature_proj, axis=-1, keepdims=True)
        feature_cos_sim = jnp.sum(x_feature_norm * feature_proj_norm, axis=-1)

        repa_loss = self.interface.mean_flat(-feature_cos_sim)

        return diffusion_loss, repa_loss

    def pred(self, *args, **kwargs):
        return self.interface.pred(*args, **kwargs)
    
    def score(self, *args, **kwargs):
        return self.interface.score(*args, **kwargs)

    def __call__(self, x: jnp.ndarray, x_feature: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        diffusion_loss, repa_loss = self.loss(x, x_feature, *args, **kwargs)

        return {
            'loss': diffusion_loss + self.repa_loss_weight * repa_loss,
            'diffusion_loss': diffusion_loss,
            'repa_loss': repa_loss
        }

        
            
