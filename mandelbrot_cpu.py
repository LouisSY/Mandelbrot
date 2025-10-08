import numpy as np
from time import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def simple_mandelbrot(width, height, real_low, rea_high, imag_low, imag_high, max_iters):
    real_vals = np.linspace(real_low, rea_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)

    # we represent members in the set with 1, and non-members as 0
    mandelbrot_set = np.ones((height, width), dtype=np.float64)

    for x in range(width):
        for y in range(height):
            c = np.complex64(real_vals[x] + 1j * imag_vals[y])
            z = np.complex64(0)
            for i in range(max_iters):
                z = z * z + c
                if np.abs(z) > 2:
                    mandelbrot_set[y, x] = 0
                    break
    return mandelbrot_set


if __name__ == "__main__":
    t1 = time()
    mandelbrot_set = simple_mandelbrot(
        width=512, height=512, real_low=-2, rea_high=2, imag_low=-2, imag_high=2, max_iters=20
    )
    t2 = time()
    mandel_time = t2 - t1
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandelbrot_set, extent=(-2, 2, -2, 2))
    plt.savefig("mandelbrot.png", dpi=fig.dpi)
    t2 = time()
    dump_time = t2 - t1
    print(f"It took {mandel_time:.4f} seconds to compute the Mandelbrot set")
    print(f"It took {dump_time:.4f} seconds to dump the image to a file")

