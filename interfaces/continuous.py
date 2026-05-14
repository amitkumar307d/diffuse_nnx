"""File containing the interface for diffusion / flow matching."""

# built-in libs
from abc import ABC, abstractmethod
import dataclasses
from enum import Enum
import math

# external libs
import flax
from flax import nnx

import jax
import jax.numpy as jnp


class TrainingTimeDistType(Enum):
    """Class for Training Time Distribution Types."""
    UNIFORM = 1
    LOGNORMAL = 2
    LOGITNORMAL = 3

    # TODO: Add more training time distribution types


class Interfaces(nnx.Module, ABC):
    r"""Base class for all diffusion / flow matching interfaces.
    
    All interfaces be a wrapper around network backbone and should support:
    - Define the pre-conditionings (see EDM)
    - Calculate losses for training
        - Define transport path (\alpha_t & \sigma_t)
        - Sample t
        - Sample X_t
    - Give tangent for sampling

    Required RNG Key:
    - 
    """

    def __init__(self, network: nnx.Module, train_time_dist_type: str | TrainingTimeDistType):
        self.network = network
        if isinstance(train_time_dist_type, str):
            self.train_time_dist_type = TrainingTimeDistType[train_time_dist_type.replace('_', '').upper()]
        else:
            self.train_time_dist_type = train_time_dist_type
    
    @abstractmethod
    def c_in(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate c_in for the interface.
        
        Args:
        - t: current timestep.

        Return:
        - c_in: c_in for the interface.
        """
    
    @abstractmethod
    def c_out(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate c_out for the interface.
        
        Args:
        - t: current timestep.

        Return:
        - c_out: c_out for the interface.
        """
    
    @abstractmethod
    def c_skip(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate c_skip for the interface.
        
        Args:
        - t: current timestep.

        Return:
        - c_skip: c_skip for the interface.
        """
    
    @abstractmethod
    def c_noise(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate c_noise for the interface.
        
        Args:
        - t: current timestep.

        Return:
        - c_noise: c_noise for the interface.
        """

    @abstractmethod
    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        r"""Sample t from the training time distribution.
        
        Args:
        - shape: shape of timestep t.
        
        Return:
        - t: sampled timestep t.
        """
    
    @abstractmethod
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        r"""Sample noises.
        
        Args:
        - shape: shape of noise.

        Return:
        - n: sampled noise.
        """
        # Exposing this function to the interface allows for more flexibility in noise sampling

    @abstractmethod
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Sample X_t according to the defined interface.
        
        Args:
        - x: input clean sample.
        - n: noise.
        - t: current timestep.

        Return:
        - x_t: Sampled X_t according to tranport path.
        """

    @abstractmethod
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Get training target.
        
        Args:
        - x: input clean sample.
        - n: noise.
        - t: current timestep.

        Return:
        - target: training target.
        """

    @abstractmethod
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Predict ODE tangent according to the defined interface.
        
        Args:
        - x_t: input noisy sample.
        - t: current timestep.
        """
    
    @abstractmethod
    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Transform ODE tangent to the Score Function \nabla \log p_t(x).
        
        Args:
        - x_t: input noisy sample.
        - t: current timestep.
        """
    
    @abstractmethod
    def loss(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Calculate loss for training.
        
        Args:
        - x: input clean sample.
        - args: additional arguments for network forward.
        - kwargs: additional keyword arguments for network forward.

        Return:
        - loss: calculated loss.
        """

    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.loss(x, *args, **kwargs)
    
    ########## Helper Functions ##########
    @staticmethod
    def mean_flat(x: jnp.ndarray) -> jnp.ndarray:
        r"""Take mean w.r.t. all dimensions of x except the first."""
        return jnp.mean(x, axis=list(range(1, x.ndim)))
    
    @staticmethod
    def bcast_right(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""Broadcast x to the right to match the shape of y."""
        assert len(y.shape) >= x.ndim
        return x.reshape(x.shape + (1,) * (len(y.shape) - x.ndim))
    
    @staticmethod
    def t_shift(t: jnp.ndarray, shift: float) -> jnp.ndarray:
        r"""Shift t by a constant shift value."""
        return shift * t / (1 + (shift - 1) * t)


class SiTInterface(Interfaces):
    r"""Interface for SiT.
    
    Transport path:
    - x_t = (1 - t) * x + t * n

    Losses:
    - L = |D - (x - n)|^2

    Predictions:
    - x = xt + t * D
    """

    def __init__(
        self, network: nnx.Module, train_time_dist_type:  str | TrainingTimeDistType,
        t_mu: float = 0., t_sigma: float = 1.0, n_mu: float = 0., n_sigma: float = 1.0, x_sigma: float = 0.5,
        t_shift_base: int = 4096,
    ):
        super().__init__(network, train_time_dist_type)
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.n_mu = n_mu
        self.n_sigma = n_sigma
        self.x_sigma = x_sigma
        self.t_shift_base = t_shift_base

    def c_in(self, t: jnp.ndarray) -> jnp.ndarray:
        # return 1 / jnp.sqrt((1 - t) ** 2 * self.x_sigma ** 2 + t ** 2)
        return jnp.ones_like(t)
    
    def c_out(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(t)
    
    def c_skip(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(t)

    def c_noise(self, t: jnp.ndarray) -> jnp.ndarray:
        return t

    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.network.rngs.time()

        if self.train_time_dist_type == TrainingTimeDistType.UNIFORM:
            t = jax.random.uniform(rng, shape=shape)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGITNORMAL:
            t = jax.nn.sigmoid(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        else:
            raise ValueError(f"Training Time Distribution Type {self.train_time_dist_type} not supported.")
        
        # shift_ratio = math.sqrt(16 * 16 * 768 / self.t_shift_base)
        # return shift_ratio * t / (1 + (shift_ratio - 1) * t)
        return t
    
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        # rng = self.make_rng('noise')
        rng = self.network.rngs.noise()

        return jax.random.normal(rng, shape=shape) * self.n_sigma + self.n_mu
    
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t = self.bcast_right(t, x)
        return (1 - t) * x + t * n
    
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return n - x
    
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t), t, *args, **kwargs
        )[0]
    
    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        tangent = self.pred(x_t, t, *args, **kwargs)
        t = self.bcast_right(t, x_t)
        return -(x_t + (1 - t) * tangent) / t
    
    def loss(self, x: jnp.ndarray, *args, return_aux=False, **kwargs) -> jnp.ndarray:
        t = self.sample_t((x.shape[0],))
        t = self.t_shift(t, math.sqrt(math.prod(x.shape[1:]) / self.t_shift_base))

        n = self.sample_n(x.shape)

        x_t = self.sample_x_t(x, n, t)
        target = self.target(x, n, t)

        net_out, features = self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t), t, *args, **kwargs
        )

        if return_aux:
            # specifically for auxiliary loss wrappers
            return self.mean_flat((net_out - target) ** 2), net_out, features
        else:
            return {
                'loss': self.mean_flat((net_out - target) ** 2)
            }


class EDMInterface(Interfaces):
    r"""Interface for EDM.
    
    Transport Path:
    - x_t = x + \sigma * n
    
    Losses:
    - L - |D - x| ^ 2

    Predictions:
    - x = D
    """

    def __init__(
        self, network: nnx.Module, train_time_dist_type:  str | TrainingTimeDistType,
        t_mu: float = 0., t_sigma: float = 1.0, n_mu: float = 0., n_sigma: float = 1.0, x_sigma: float = 0.5
    ):
        super().__init__(network, train_time_dist_type)
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.n_mu = n_mu
        self.n_sigma = n_sigma
        self.x_sigma = x_sigma

    def c_in(self, t: jnp.ndarray) -> jnp.ndarray:
        return 1 / jnp.sqrt(self.x_sigma ** 2 + t ** 2)
    
    def c_out(self, t: jnp.ndarray) -> jnp.ndarray:
        return t * self.x_sigma / jnp.sqrt(t ** 2 + self.x_sigma ** 2)
    
    def c_skip(self, t) -> jnp.ndarray:
        return self.x_sigma ** 2 / (t ** 2 + self.x_sigma ** 2)

    def c_noise(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(t) / 4

    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.network.rngs.time()
        if self.train_time_dist_type == TrainingTimeDistType.UNIFORM:
            return jax.random.uniform(rng, shape=shape)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGNORMAL:
            return jnp.exp(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGITNORMAL:
            return jax.nn.sigmoid(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        else:
            raise ValueError(f"Training Time Distribution Type {self.train_time_dist_type} not supported.")
    
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.network.rngs.noise()

        return jax.random.normal(rng, shape=shape) * self.n_sigma + self.n_mu
    
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return x + self.bcast_right(t, n) * n
    
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return x
    
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:

        F_x = self.network((self.bcast_right(self.c_in(t), x_t) * x_t), self.c_noise(t), *args, **kwargs)[0]
        D_x = self.bcast_right(self.c_skip(t), x_t) * x_t + self.bcast_right(self.c_out(t), F_x) * F_x

        return (x_t - D_x) / self.bcast_right(t, x_t)
    
    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        tangent = self.pred(x_t, t, *args, **kwargs)
        t = self.bcast_right(t, x_t)
        return -(x_t - tangent) / (t ** 2)
    
    def loss(self, x: jnp.ndarray, *args, return_aux=False, **kwargs) -> jnp.ndarray:
        sigma = self.sample_t((x.shape[0],))
        n = self.sample_n(x.shape)

        x_t = self.sample_x_t(x, n, sigma)
        target = self.target(x, n, sigma)

        F_x, features = self.network((self.bcast_right(self.c_in(sigma), x_t) * x_t), self.c_noise(sigma), *args, **kwargs)
        D_x = self.bcast_right(self.c_skip(sigma), x_t) * x_t + self.bcast_right(self.c_out(sigma), F_x) * F_x

        weight = (sigma ** 2 + self.x_sigma ** 2) / (sigma * self.x_sigma) ** 2

        if return_aux:
            # specifically for auxiliary loss wrappers
            return self.mean_flat(self.bcast_right(weight, D_x) * (D_x - target) ** 2), D_x, features
        else:
            return {
                'loss': self.mean_flat(self.bcast_right(weight, D_x) * (D_x - target) ** 2)
            }


class sCTInterface(EDMInterface):
    r"""Interface for CM.
    
    Transport Path:
    - x_t = x + \sigma * n
    
    Losses:
    - L - |f_{t - 1} - f_{t}| ^ 2

    Predictions:
    - x = D
    """

    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Get the effective training target for sCT."""
        pass

    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """Predict the average velocity from noise to data."""
        pass

    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise ValueError("sCTInterface does not support score calculation.")
    
    def loss(self, x: jnp.ndarray, *args, return_aux=False, **kwargs) -> jnp.ndarray:
        """Calculate the sCT loss."""
        pass


class sCDInterface(sCTInterface):
    r"""Interface for CM.
    
    Transport Path:
    - x_t = x + \sigma * n
    
    Losses:
    - L - |f_{t - 1} - f_{t}| ^ 2

    Predictions:
    - x = D
    """

    def __init__(
        self, network: nnx.Module, train_time_dist_type:  str | TrainingTimeDistType,
        t_mu: float = 0., t_sigma: float = 1.0, n_mu: float = 0., n_sigma: float = 1.0, x_sigma: float = 0.5,
        teacher: nnx.Module | None = None, guidance_scale: float = 1.0
    ):
        assert teacher is not None, "Teacher model must be provided for sCDInterface."
        super().__init__(
            network,
            train_time_dist_type,
            t_mu=t_mu, t_sigma=t_sigma, n_mu=n_mu, n_sigma=n_sigma, x_sigma=x_sigma
        )
        self.teacher = teacher
        self.guidance_scale = guidance_scale

    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise ValueError("sCDInterface does not support score calculation.")

    def loss(self, x: jnp.ndarray, *args, return_aux=False, **kwargs) -> jnp.ndarray:
        """Calculate the sCD loss."""
        pass


class MeanFlowInterface(SiTInterface):
    r"""Interface for Mean Flow.
    
    Transport Path:
    - x_t = (1 - t) * x + t * n

    Losses:
    - L = |u - sg(dx/dt - v du/dx - du/dt)|^2

    Predictions:
    - x = xt + t * D
    """

    def __init__(
        self, network: nnx.Module, train_time_dist_type:  str | TrainingTimeDistType,
        t_mu: float = 0., t_sigma: float = 1.0, n_mu: float = 0., n_sigma: float = 1.0, x_sigma: float = 0.5,
        guidance_scale: float = 1.0, guidance_mixture_ratio: float = 0.5, guidance_t_min: float = 0.0, guidance_t_max: float = 1.0,
        norm_eps: float = 1e-3, norm_power: float = 1.0, fm_portion: float = 0.75, cond_drop_ratio: float = 0.5,
        t_shift_base: int = 4096,
    ):
        super().__init__(
            network,
            train_time_dist_type,
            t_mu=t_mu, t_sigma=t_sigma, n_mu=n_mu, n_sigma=n_sigma, x_sigma=x_sigma
        )
        # omega in meanflow
        self.guidance_scale = guidance_scale
        # keppa in meanflow
        self.guidance_mixture_ratio = guidance_mixture_ratio

        # effectively guidance interval
        self.guidance_t_min = guidance_t_min
        self.guidance_t_max = guidance_t_max

        self.norm_eps = norm_eps
        self.norm_power = norm_power
        self.fm_portion = fm_portion
        self.cond_drop_ratio = cond_drop_ratio

    def sample_t_r(self, shape: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
        t = self.sample_t(shape)
        r = self.sample_t(shape)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        fm_mask = jnp.arange(t.shape[0]) < int(t.shape[0] * self.fm_portion)
        r = jnp.where(fm_mask, t, r)
        return t, r

    def cond_drop(self, x: jnp.ndarray, n: jnp.ndarray, v: jnp.ndarray, y: jnp.ndarray, neg_y: jnp.ndarray) -> jnp.ndarray:
        """Drop the condition with a certain ratio.
        
        Note: the reason why we need to drop the condition outside of the model is that
              the effective regression target depends on the resulted from dropout insta velocity
        """
        unguided_v = n - x

        mask = jax.random.uniform(self.network.rngs.label_dropout(), shape=y.shape) < self.cond_drop_ratio
        num_drop = jnp.sum(mask).astype(jnp.int32)
        drop_mask = jnp.arange(y.shape[0]) < num_drop
        # TODO: consider supporting more generalized negative condition
        y = jnp.where(drop_mask, neg_y, y)
        v = jnp.where(self.bcast_right(drop_mask, v), unguided_v, v)

        return v, y

    def insta_velocity(
        self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray, *args,
        y: jnp.ndarray | None = None, neg_y: jnp.ndarray | None = None, **kwargs
    ) -> jnp.ndarray:
        """Instantaneous velocity of the mean flow."""
        self.network.eval()
        v = n - x

        x_t = self.sample_x_t(x, n, t)

        # TODO: fix the hardcoding
        # unconditional generation
        if y is None:
            y = jnp.zeros((t.shape[0],), dtype=jnp.int32) + 1000
        # default negative condition
        if neg_y is None:
            neg_y = jnp.zeros((t.shape[0],), dtype=jnp.int32) + 1000

        # no guidance case
        if self.guidance_scale == 1.0 and self.guidance_mixture_ratio == 0.0:
            return v, y, neg_y

        v_uncond = self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t),
            t,
            *args,
            y=neg_y,
            dt=jnp.zeros_like(t),
            **kwargs
        )[0]
        if self.guidance_mixture_ratio == 0.0:
            return jnp.where(
                self.bcast_right((t >= self.guidance_t_min) & (t <= self.guidance_t_max), v),
                v_uncond + self.guidance_scale * (v - v_uncond),
                v
            ), y, neg_y
        
        v_cond = self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t),
            t,
            *args,
            y=y,
            dt=jnp.zeros_like(t),
            **kwargs
        )[0]

        self.network.train()
        return jnp.where(
            self.bcast_right((t >= self.guidance_t_min) & (t <= self.guidance_t_max), v),
            self.guidance_scale * v + (1 - self.guidance_scale - self.guidance_mixture_ratio) * v_uncond + self.guidance_mixture_ratio * v_cond,
            v
        ), y, neg_y

    def target(
        self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray, r: jnp.ndarray, *args,
        y: jnp.ndarray | None = None, neg_y: jnp.ndarray | None = None, **kwargs
    ) -> jnp.ndarray:
        r"""Get training target for Mean Flow.
        
        Note: network must be augmented with r, the jump size, as an additional input
        """
        v, y, neg_y = self.insta_velocity(x, n, t, *args, y=y, neg_y=neg_y, **kwargs)
        v, y = self.cond_drop(x, n, v, y, neg_y=neg_y)
        x_t = self.sample_x_t(x, n, t)

        def u_fn(x_t, t, r):
            return self.network(
                (self.bcast_right(self.c_in(t), x_t) * x_t),
                t,
                *args,
                dt=t - r,
                y=y,
                **kwargs
            )
        
        dtdt = jnp.ones_like(t)
        drdt = jnp.zeros_like(r)
        u, dudt, feat = jax.jvp(
            u_fn,
            (x_t, t, r),
            (v, dtdt, drdt),
            has_aux=True
        )

        return (u, feat), jax.lax.stop_gradient(v - jnp.clip(self.bcast_right(t - r, v), 0., 1.) * dudt)

    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, r: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Predict ODE tangent according to the Mean Flow interface."""
        return self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t),
            t,
            *args,
            dt=t - r,
            **kwargs
        )[0]

    def score(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        # score is given at r = t
        tangent = self.pred(x_t, t, jnp.zeros_like(t), *args, **kwargs)
        t = self.bcast_right(t, x_t)
        return -(x_t + (1 - t) * tangent) / t ** 2

    def loss(self, x: jnp.ndarray, *args, return_aux=False, **kwargs) -> jnp.ndarray:
        
        t, r = self.sample_t_r((x.shape[0],))
        n = self.sample_n(x.shape)

        (net_out, features), target = self.target(x, n, t, r, *args, **kwargs)

        # following the implementation of meanflow we use sum loss
        loss = jnp.sum((net_out - target) ** 2, axis=list(range(1, x.ndim)))
        adp_w = 1.0 / (loss + self.norm_eps) ** self.norm_power
        loss = jax.lax.stop_gradient(adp_w) * loss
        
        if return_aux:
            return loss, net_out, features
        else:
            return {
                'loss': loss
            }

        

    
