import torch
from torch.nn import functional as F


def compute_vendi_score(X, q=1, normalize=True, kernel="linear", device="cuda"):
    X = X.to(device)
    if normalize:
        X = F.normalize(X, p=2, dim=1)
    n = X.shape[0]
    if kernel == "linear":
        S = torch.mm(X, X.t())
    elif kernel == "polynomial":
        S = (torch.mm(X, X.t()) + 1) ** 3  # Polynomial kernel with degree=3, coef0=1
    else:
        raise NotImplementedError("kernel not implemented")

    w = torch.linalg.eigvalsh(S / n)
    return torch.exp(entropy_q(w, q=q)).item()


def entropy_q(p, q=1):
    p = p[p > 0]
    if q == 1:
        return -(p * torch.log(p)).sum()
    if q == float("inf"):
        return -torch.log(torch.max(p))
    return torch.log((p**q).sum()) / (1 - q)


def compute_per_class_vendi_scores(reps, labels):
    num_classes = len(torch.unique(labels))
    vendi_per_class = torch.zeros(num_classes, device=reps.device)
    for i in range(num_classes):
        reps_class = reps[labels == i]
        vendi_per_class[i] = compute_vendi_score(reps_class)
    return vendi_per_class


# Example usage
if __name__ == "__main__":
    # Assuming reps is a PyTorch tensor of shape (N, D) where N is the number of samples and D is the feature dimension
    # labels is a PyTorch tensor of shape (N,) containing class labels
    reps = torch.randn(1000, 128).cuda()
    labels = torch.randint(0, 1, (1000,)).cuda()

    vendi_score = compute_vendi_score(reps)
    print(f"Overall Vendi Score: {vendi_score:.4f}")

    per_class_scores = compute_per_class_vendi_scores(reps, labels)
    print("Per-class Vendi Scores:")
    for i, score in enumerate(per_class_scores):
        print(f"Class {i}: {score.item():.4f}")
