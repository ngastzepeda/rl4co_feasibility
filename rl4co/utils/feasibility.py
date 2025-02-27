import torch


def tensor_nan_to_inf(tensor):
    return torch.where(
        torch.isnan(tensor),
        torch.tensor(
            float("inf"),
            device=tensor.device,
            dtype=tensor.dtype,
        ),
        tensor,
    )


def tensor_nan_to_neg_inf(tensor):
    return torch.where(
        torch.isnan(tensor),
        torch.tensor(
            float("-inf"),
            device=tensor.device,
            dtype=tensor.dtype,
        ),
        tensor,
    )


def tensor_inf_to_nan(tensor):
    return torch.where(
        tensor == float("inf"),
        torch.tensor(
            float("nan"),
            device=tensor.device,
            dtype=tensor.dtype,
        ),
        tensor,
    )


def tensor_neg_inf_to_nan(tensor):
    return torch.where(
        tensor == float("-inf"),
        torch.tensor(
            float("nan"),
            device=tensor.device,
            dtype=tensor.dtype,
        ),
        tensor,
    )
