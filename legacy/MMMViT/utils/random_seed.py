import os
import random
import numpy as np
import torch

def setup_seed(seed):
    # Set Python hashing seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set random seeds for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        if num_gpus <= 1:
            torch.cuda.manual_seed(seed)
        else: #multi-GPUs
            torch.cuda.manual_seed_all(seed) 
    else:
        print("No GPUs available!")

    # # Enforce deterministic algorithms in PyTorch
    # torch.use_deterministic_algorithms(True)

    # Disable CuDNN benchmarking and enable deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True