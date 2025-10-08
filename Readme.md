# Mandelbrot Set Generator

A Python implementation for generating and visualizing the Mandelbrot set, exploring both CPU and GPU computation approaches with performance profiling.

## Overview

The Mandelbrot set is a famous fractal defined by the iterative formula `z(n+1) = z(n)² + c`, where `c` is a complex number representing a point in the complex plane. This project generates visual representations of the Mandelbrot set and serves as a platform for exploring GPU acceleration techniques with detailed performance analysis.

## Features

- CPU-based Mandelbrot set computation with NumPy
- Built-in performance timing and benchmarking
- Detailed profiling support for optimization analysis
- PNG output with matplotlib visualization
- Configurable resolution, iteration limits, and complex plane boundaries
- Foundation for GPU acceleration experiments

## Requirements

- Python 3.13
- numpy
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LouisSY/Mandelbrot.git
cd Mandelbrot
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage
Run the CPU implementation:
```bash
python mandelbrot_cpu.py
```

### Performance Profiling
Generate detailed performance profile:
```bash
python -m cProfile -s cumtime mandelbrot_cpu.py > mandelbrot_profile.txt
```

This creates a comprehensive performance report showing:
- Function call counts
- Total time spent in each function
- Cumulative time including sub-calls
- Per-call timing statistics

View the profile results:
```bash
cat mandelbrot_profile.txt
```

## Configuration

Modify parameters in `simple_mandelbrot()`:

- `width`, `height`: Image resolution in pixels (512×512 default)
- `real_low`, `rea_high`: Real axis boundaries (-2 to 2 default)
- `imag_low`, `imag_high`: Imaginary axis boundaries (-2 to 2 default)
- `max_iters`: Maximum iterations before considering convergence (20 default)

## Algorithm

The CPU implementation uses a pixel-by-pixel approach:

1. Generate coordinate grids for real and imaginary axes using `np.linspace()`
2. For each pixel, convert coordinates to complex number `c`
3. Iterate `z = z² + c` starting from `z = 0`
4. If `|z| > 2`, the point diverges (set to 0)
5. Points remaining bounded are part of the set (remain 1)

## Performance Analysis

### Current CPU Performance
Typical performance on modern hardware:
- 512×512 resolution: ~2-5 seconds computation
- Computation time scales O(width × height × max_iters)
- Memory usage: O(width × height) for result array

### Profiling Insights
Use the generated `mandelbrot_profile.txt` to identify:
- Bottlenecks in the nested loops
- NumPy operation costs
- Memory allocation patterns
- Function call overhead

## GPU Acceleration (Planned)

This project aims to explore GPU acceleration using:

### macOS (Apple Silicon)
- **PyTorch with Metal**: Leverage Apple's Metal Performance Shaders

### Linux/Windows
- **PyCUDA**: Direct CUDA programming for NVIDIA GPUs


## Output Files

The script generates:
- `mandelbrot.png`: Visual representation of the Mandelbrot set
- `mandelbrot_profile.txt`: Detailed performance profile (when using cProfile)
- Console output with timing metrics

## Project Structure

```
Mandelbrot/
├── mandelbrot_cpu.py           # CPU implementation
├── mandelbrot.png              # Generated fractal image
├── mandelbrot_profile.txt      # Performance profile (optional)
├── README.md                   # This file
└── requirements.txt            # Dependencies (optional)
```


## Future Enhancements

- [ ] GPU-accelerated versions (PyTorch Metal, PyCUDA, PyOpenCL)
- [ ] Vectorized NumPy implementation
- [ ] Interactive zoom and pan functionality
- [ ] Color mapping for iteration counts
- [ ] Performance comparison benchmarks
- [ ] Different fractal algorithms (Julia sets, Burning Ship)
- [ ] Multi-threading CPU implementation

## Profiling and Benchmarking

For developers exploring optimization:

```bash
# Basic profiling
python -m cProfile -s cumtime mandelbrot_cpu.py > mandelbrot_profile.txt

```

## Contributing

This project serves as a learning platform for GPU computing and performance optimization. Contributions welcome, especially:
- GPU acceleration implementations
- Performance optimizations
- Profiling and benchmarking tools
- Visualization improvements

## License

MIT License

## References

- [Mandelbrot Set - Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Python Profiling Documentation](https://docs.python.org/3/library/profile.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/basics.performance.html)
