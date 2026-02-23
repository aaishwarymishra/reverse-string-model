from ignite import metrics
import torch 

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

def loss_fn(preds, y):
    # Permute predictions to match CrossEntropyLoss expected input format (batch, dim, seq)
    preds = preds.permute(0, 2, 1)
    loss = criterion(preds, y)
    return loss

def accuracy_transform(output):
    preds, y = output

    preds_flat = preds.reshape(-1, preds.size(-1))
    y_flat = y.reshape(-1)

    mask = y_flat != 0

    preds_masked = preds_flat[mask]
    y_masked = y_flat[mask]

    return preds_masked, y_masked

def get_metrics():
    acc_metric = metrics.Accuracy(output_transform=accuracy_transform)
    return {"accuracy": acc_metric,"loss": metrics.Loss(loss_fn)}

