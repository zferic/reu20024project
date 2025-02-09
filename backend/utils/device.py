import torch




def get_device():
    """
    Uses torch to return the name of the best device to use.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"
        

