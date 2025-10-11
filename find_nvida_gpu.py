import pycuda.driver as drv

drv.init()
print(f"Detected {drv.Device.count()} CUDA device(s).")

cuda_cores_per_mp = {
    # Maxwell
    (5, 0): 128,
    (5, 2): 128,
    (5, 3): 128,
    # Pascal
    (6, 0): 64,
    (6, 1): 128,
    (6, 2): 128,
    # Volta
    (7, 0): 64,
    (7, 2): 64,
    # Turing
    (7, 5): 64,
    # Ampere
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    # Ada Lovelace
    (8, 9): 128,
    # Hopper
    (9, 0): 128,
    # Blackwell
    (10, 0): 128,
}

for i in range(drv.Device.count()):
    gpu_device = drv.Device(i)
    print(f"Device {i}: {gpu_device.name()}")
    compute_capability = gpu_device.compute_capability()
    print(f"Compute capability: {compute_capability}")
    print(f"Device memory size: {gpu_device.total_memory() // (1024 ** 2)} MB")
    device_attributes = {str(k): v for k, v in gpu_device.get_attributes().items()}
    num_mp = device_attributes["MULTIPROCESSOR_COUNT"]
    cores_per_mp = cuda_cores_per_mp.get(compute_capability, "Unknown")
    total_cores = num_mp * cores_per_mp if isinstance(cores_per_mp, int) else "Unknown"
    print(f"Number of multiprocessors: {num_mp}, CUDA cores per MP: {cores_per_mp}, Total CUDA Cores: {total_cores}")
    device_attributes.pop("MULTIPROCESSOR_COUNT")
    for k, v in device_attributes.items():
        print(f"{k}: {v}")
