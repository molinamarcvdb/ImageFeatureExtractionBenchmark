import torch


def sw_approx(X, Y, device="cuda"):
    """
    Approximate Sliced W2 without Monte Carlo
    From https://arxiv.org/pdf/2106.15427.pdf

    Args:
    X, Y: PyTorch tensors of shape (n_samples, n_features)
    device: The device to perform computations on ('cuda' or 'cpu')

    Returns:
    approx_sw: Approximated Sliced Wasserstein distance
    """
    X = X.to(device)
    Y = Y.to(device)

    d = X.shape[1]
    mean_X = X.mean(dim=0)
    mean_Y = Y.mean(dim=0)

    mean_term = torch.norm(mean_X - mean_Y) ** 2 / d
    m2_Xc = (torch.norm(X - mean_X, dim=1) ** 2).mean() / d
    m2_Yc = (torch.norm(Y - mean_Y, dim=1) ** 2).mean() / d

    approx_sw = (mean_term + (m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2)) ** 2) ** (1 / 2)

    return approx_sw
