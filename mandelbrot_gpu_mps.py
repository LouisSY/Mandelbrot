import torch
import numpy as np
from time import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def metal_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters):
    # Use Metal Performance Shaders on Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create coordinate arrays on Metal GPU
    real_vals = torch.linspace(real_low, real_high, width, device=device)
    imag_vals = torch.linspace(imag_low, imag_high, height, device=device)

    # Create meshgrid on GPU
    real_grid, imag_grid = torch.meshgrid(real_vals, imag_vals, indexing='xy')
    c = torch.complex(real_grid, imag_grid)

    # Initialize arrays on GPU
    z = torch.zeros_like(c)
    mandelbrot_set = torch.ones((height, width), dtype=torch.float32, device=device)

    for i in range(max_iters):
        # Vectorized operations on Metal GPU
        mask = torch.abs(z) <= 2
        z = torch.where(mask, z * z + c, z)

        # Mark diverged points
        diverged = (torch.abs(z) > 2) & (mandelbrot_set == 1)
        mandelbrot_set[diverged] = 0

    # Transfer result back to CPU
    return mandelbrot_set.cpu().numpy()


if __name__ == "__main__":
    t1 = time()
    mandelbrot_set = metal_mandelbrot(
        width=512, height=512, real_low=-2, real_high=2, imag_low=-2, imag_high=2, max_iters=20
    )
    t2 = time()
    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandelbrot_set, extent=(-2, 2, -2, 2))
    plt.savefig("mandelbrot_metal.png", dpi=fig.dpi)
    t2 = time()
    dump_time = t2 - t1

    print(f"It took {mandel_time:.4f} seconds to compute the Mandelbrot set on Metal GPU")
    print(f"It took {dump_time:.4f} seconds to dump the image to a file")
