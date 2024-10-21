import torch


def compute_authpct(train_feat, gen_feat, device="cuda"):
    # Ensure inputs are PyTorch tensors with gradients enabled
    train_feat = torch.tensor(
        train_feat, dtype=torch.float32, device=device, requires_grad=True
    )
    gen_feat = torch.tensor(
        gen_feat, dtype=torch.float32, device=device, requires_grad=True
    )

    real_dists = torch.cdist(train_feat, train_feat)

    # Use a large value instead of infinity to avoid potential numerical issues
    real_dists.fill_diagonal_(1e10)
    gen_dists = torch.cdist(train_feat, gen_feat)

    real_min_dists, real_min_indices = real_dists.min(axis=0)
    gen_min_dists, gen_min_indices = gen_dists.min(dim=0)

    # For every synthetic point, find its closest real point, d1
    # Then, for that real point, find its closest real point(not itself), d2
    # if d2<d1, then it's authentic
    authen = real_min_dists[gen_min_indices] < gen_min_dists
    authpct = 100 * torch.sum(authen.float()) / len(authen)

    return authpct


# Example usage
if __name__ == "__main__":
    # Simulate some feature data
    train_feat = torch.randn(1000, 128, requires_grad=True)
    gen_feat = torch.randn(500, 128, requires_grad=True)

    # Compute AuthPct
    authpct = compute_authpct(train_feat, gen_feat)
    print(f"AuthPct: {authpct.item():.2f}%")

    # Backpropagation example
    authpct.backward()

    # Check if gradients were computed
    print("Gradient for train_feat:", train_feat.grad is not None)
    print("Gradient for gen_feat:", gen_feat.grad is not None)
