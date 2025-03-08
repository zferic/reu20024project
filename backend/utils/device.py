import torch
import os
import logging
import psutil

logger = logging.getLogger(__name__)

def get_device():
    """
    Uses torch to return the name of the best device to use.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"

def get_cpu_info():
    """
    Returns information about the CPU for optimization purposes.
    """
    cpu_info = {
        "cpu_count": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    
    logger.info(f"CPU Info: {cpu_info}")
    return cpu_info

def optimize_for_cpu():
    """
    Apply optimizations for CPU-based environments.
    """
    cpu_info = get_cpu_info()
    
    # Set PyTorch thread settings
    if hasattr(torch, "set_num_threads"):
        # Use half of available logical cores for better parallelism without overloading
        recommended_threads = max(1, cpu_info["logical_cores"] // 2)
        torch.set_num_threads(recommended_threads)
        logger.info(f"Set PyTorch to use {recommended_threads} CPU threads")
    
    # Set inter-op parallelism threads if available
    if hasattr(torch, "set_num_interop_threads"):
        # Use physical core count for inter-op parallelism
        recommended_interop = max(1, cpu_info["physical_cores"])
        torch.set_num_interop_threads(recommended_interop)
        logger.info(f"Set PyTorch inter-op threads to {recommended_interop}")
    
    # Return optimization settings for reference
    return {
        "num_threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        "recommended_batch_size": max(1, min(8, cpu_info["physical_cores"])),
        "recommended_workers": max(1, cpu_info["physical_cores"] - 1),
    }
        

