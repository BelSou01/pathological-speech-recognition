import torch

from .metrics import classification_metrics


def evaluate_model(
    model,
    dataloader,
    device
):
    """
    Evaluate a classification model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    dataloader : DataLoader
        Test data loader.

    device : torch.device
        CPU or GPU.

    Returns
    -------
    dict
        Classification metrics.
    """

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.numpy()
            )

    results = classification_metrics(
        all_targets,
        all_predictions
    )

    return results
