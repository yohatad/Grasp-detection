import logging
import torch

logging.basicConfig(level=logging.INFO)

# Check if CUDA can be used
def get_device(force_cpu):
    if torch.cuda.is_available() and not force_cpu:
        logging.info("With GPU acceleration (CUDA found).")
        device = torch.device("cuda")
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device
