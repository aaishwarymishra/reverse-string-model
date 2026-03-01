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

    def exact_match_transform(output):
        preds, y = output
        # preds: (B, L, C), y: (B, L)
        pred_indices = preds.argmax(dim=-1)

        # Exact match per sequence
        mask = y != pad_idx

        # Check if all non-pad tokens match exactly
        correct_per_token = pred_indices == y
        # Only care about non-padding tokens matching:
        # For a sequence to be perfectly matched, all its meaningful tokens must match
        # (correct_per_token | ~mask) means either true match or it's pad so we ignore
        is_exact_match = (correct_per_token | ~mask).all(dim=-1)

        # Format for Accuracy metric: (preds, targets)
        # We want to measure the average of `is_exact_match`.
        # We can simulate this by passing y_pred=is_exact_match and y=ones.
        return is_exact_match.long(), torch.ones_like(is_exact_match, dtype=torch.long)

    return {
        "accuracy": metrics.Accuracy(output_transform=accuracy_transform),
        "exact_match": metrics.Accuracy(output_transform=exact_match_transform),
        "loss": metrics.Loss(loss_fn),
    }
