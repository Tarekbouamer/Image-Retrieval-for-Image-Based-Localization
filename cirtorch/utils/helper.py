import torch

__all__ = [
    "_extract_device_dtype"
]

def _extract_device_dtype(tensor_list):
    """
        Check if all the input are in the same device (only if when they are torch.Tensor).
        If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).
    """

    device, dtype = None, None

    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                continue

            _device = tensor.device
            _dtype = tensor.dtype

            if device is None and dtype is None:
                device = _device
                dtype = _dtype

            elif device != _device or dtype != _dtype:
                raise ValueError("Passed values are not in the same device and dtype."
                                 f"Got ({device}, {dtype}) and ({_device}, {_dtype}).")

    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    return (device, dtype)