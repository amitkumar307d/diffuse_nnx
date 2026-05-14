import math
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == "__main__":

    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    loader_iter = iter(loader)
    data = trainset[0][0].numpy()
    rng = jax.random.PRNGKey(42)
    n = jax.random.normal(rng, data.shape)
    num_data = len(trainset)
    ys = [
        data[0].flatten().numpy() for data in trainset
    ]
    ys = jnp.stack(ys)

    ##### Get noising kernel
    @jax.jit
    def p(x, t):
        k = 32 * 32 * 3
        
        diff = x[:, None, ...] - ys
        # print(diff.shape)
        kernel = jnp.sum(jnp.exp(-jnp.mean(diff ** 2, axis=-1) / (2 * t ** 2)), axis=1)
        return kernel / num_data
    

    # prob = p(data.flatten() + 80 * n.flatten(), 80)
    # print(prob)
    # noise_level = np.linspace(0.02, 80, 100)
    step_indices = jnp.arange(35)
    sigma_min = 0.002
    sigma_max = 80.
    rho = 7.

    t_steps = (
        sigma_max ** (1 / rho)
        +
        step_indices / (35 - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    # ensure last step is 0
    # noise_level = jnp.concatenate([t_steps, jnp.array([0.])])
    noise_level = jnp.linspace(0., 1., 100)

    if not os.path.exists('entropy.npy'):
        ets = []
        for t in tqdm(noise_level, desc='calc entropy'):
            entropy = 0
            loader_iter = iter(loader)
            for data in tqdm(loader_iter):
                y = data[0].numpy()
                y = y.reshape(y.shape[0], -1)
                rng, n_rng = jax.random.split(rng)
                n = jax.random.normal(n_rng, y.shape)
                prob = p(y + t * n, t)
                entropy += np.sum(prob * np.log(prob))
            
            ets.append(entropy * -1)
        
        ets = np.stack(ets)
        np.save('entropy', ets)
    else:
        print('found pre-saved file')
        ets = np.load('entropy.npy')

    # plt.xscale('log')
    print(noise_level)
    plt.plot(noise_level, ets)
    plt.savefig('entropy.png')
    # print(p(data.flatten() + 80 * n.flatten(), 80))
    
