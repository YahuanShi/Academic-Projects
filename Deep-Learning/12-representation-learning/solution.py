import torch
import torch.nn.functional as F
import torchvision.transforms as T


def simclr_loss(z1, z2, temperature=0.5, device='cpu'):
    bs, dim_feat = z1.shape
    reps = torch.cat([z1, z2], dim=0)
    reps = F.normalize(reps, dim=-1)

    logits = torch.matmul(reps, reps.T) / temperature

    # Filter out similarities of samples with themself
    mask = torch.eye(2 * bs, dtype=torch.bool, device=device)
    logits = logits[~mask]
    logits = logits.view(2 * bs, 2 * bs - 1)  # [2*b, 2*b-1]

    # The labels point from a sample in z1 to its equivalent in z2
    labels = torch.arange(bs, dtype=torch.long, device=device)
    labels = torch.cat([labels + bs - 1, labels])

    loss = F.cross_entropy(logits, labels)
    return loss


augment = T.Compose([T.RandomResizedCrop(28, scale=(0.1, 1.0), interpolation=T.InterpolationMode('bicubic')),
                     T.RandomHorizontalFlip(),
                     T.ToTensor(),
                     T.Normalize((0.5,), (0.5,))
                     ])


def sample_two_views(x):
    x1 = augment(x)
    x2 = augment(x)
    return x1, x2


def compute_embeddings(dl, encoder, device):
    X = []
    Y = []
    with torch.no_grad():
        for x, y, in dl:
            x = x.to(device)
            z = encoder(x)
            X.append(z)
            Y.append(y)
    X = torch.cat(X).cpu()
    Y = torch.cat(Y).cpu()
    return X, Y
