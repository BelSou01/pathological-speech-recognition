import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Train a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.

    dataloader : DataLoader
        Training data loader.

    criterion : torch.nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer.

    device : torch.device
        CPU or GPU.

    Returns
    -------
    epoch_loss : float
        Average training loss.

    epoch_accuracy : float
        Training accuracy.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        total += targets.size(0)

        correct += (
            predictions == targets
        ).sum().item()

    epoch_loss = (
        running_loss / len(dataloader)
        if len(dataloader) > 0
        else 0.0
    )

    epoch_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return epoch_loss, epoch_accuracy
