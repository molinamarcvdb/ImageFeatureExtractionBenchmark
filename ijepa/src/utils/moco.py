import torch
import torch.nn as nn
import torch.distributed as dist


class MocoLoss(nn.Module):

    def __init__(self, T=1.0):
        super(MocoLoss, self).__init__()
        self.T = T

    def forward(self, q, k):
        """
        Input:
            q: queries (typically current batch) - requires gradients
            k: keys (typically from the queue) - does not require gradients
        Output:
            loss
        """
        # normalize
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            k = nn.functional.normalize(k, dim=1)
            k = concat_all_gather(k)

        # Einstein sum for batch-wise dot product
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T

        # labels: positive key indicators
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()

        # compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
