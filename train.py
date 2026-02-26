import torch
from ignite import metrics, handlers


def train_step(engine, model, batch, loss, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x, y = batch[0].to(device), batch[1].to(device)

    logits = model(x)
    loss = loss(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(engine, model, batch, device):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y


def get_loss_fn(pad_idx):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    def loss_fn(preds, y):
        preds = preds.permute(0, 2, 1)
        loss = criterion(preds, y)
        return loss

    return loss_fn


def get_metrics(loss_fn, pad_idx=0):
    def accuracy_transform(output):
        preds, y = output

        preds_flat = preds.reshape(-1, preds.size(-1))
        y_flat = y.reshape(-1)

        mask = y_flat != pad_idx

        preds_masked = preds_flat[mask]
        y_masked = y_flat[mask]

        return preds_masked, y_masked

    return {
        "accuracy": metrics.Accuracy(output_transform=accuracy_transform),
        "loss": metrics.Loss(loss_fn),
    }

    def default_log_evaluation(
        engine, train_evaluator, val_evaluator=None, train_loader=None, val_loader=None
    ):
        train_evaluator.run(train_loader)
        train_metrics = train_evaluator.state.metrics

        train_str = f"Training Results - Epoch: {engine.state.epoch}"
        for name, value in train_metrics.items():
            train_str += f" {name}: {value:.4f}"
        print(train_str)

        # Run val evaluator if provided
        if val_loader is not None and val_evaluator is not None:
            val_evaluator.run(val_loader)
            val_metrics = val_evaluator.state.metrics

            val_str = f"Validation Results - Epoch: {engine.state.epoch}"
            for name, value in val_metrics.items():
                val_str += f" {name}: {value:.4f}"
            print(val_str)
