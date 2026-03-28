import torch
def hardware():
    """Returns the best available device: CUDA, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
