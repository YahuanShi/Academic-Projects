import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def solution_1a_create_model(hidden_dims: list[int]) -> nn.Module:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, hidden_dims[0]),
        nn.ReLU(),
        *[nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU())
            for in_feats, out_feats in zip(hidden_dims[:-1], hidden_dims[1:])],
        nn.Linear(hidden_dims[-1], 10)
    )
    return model


def solution_2b_evaluate(model: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    losses = []
    accuracies = []

    model.eval()  # TODO: explain in markdown

    with torch.no_grad():  # TODO: explain in markdown
        for batch, labels in dataloader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(batch)

            loss = cross_entropy(preds, labels)
            losses.append(loss.item())

            accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels)
            accuracies.append(accuracy.item() / len(labels))

    model.train()  # TODO: explain in markdown

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)

    return avg_loss, avg_acc


def solution_2a_train_one_epoch(model, dataloader, optimizer):
    losses = []
    for batch, labels in dataloader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        preds = model(batch)
        loss = cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def solution_2c_print_loss_every_n_epochs(
        model: nn.Module,
        epoch: int,
        train_losses: list[float],
        val_losses: list[float],
        val_acc: float,
        n: int) -> None:
    if epoch % n == 0:
        print(f"[EPOCH {epoch}] Train loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}, Validation accuracy: {val_acc:.4f}")


def solution_3b_save_model_if_improved(model, epoch, train_losses, val_losses, val_acc, model_name):
    if val_losses[-1] == min(val_losses):
        print(f"Found new best model at epoch {epoch} with a validation loss of {val_losses[-1]:.4f}")
        torch.save(model.state_dict(), model_name)


def solution_4a_create_model_with_dropout(hidden_dims: list[int], p: float):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, hidden_dims[0]),
        nn.ReLU(),
        nn.Dropout(p),
        *[nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU(), nn.Dropout(p))
            for in_feats, out_feats in zip(hidden_dims[:-1], hidden_dims[1:])],
        nn.Linear(hidden_dims[-1], 10)
    )
    return model

def solution_4a_len(noise_augmented_dataset):
    return len(noise_augmented_dataset.dataset)


def solution_4b_getitem(noise_augmented_dataset, index):
    x, y = noise_augmented_dataset.dataset[index]
    x = x + noise_augmented_dataset.eps * torch.randn_like(x)
    return x, y