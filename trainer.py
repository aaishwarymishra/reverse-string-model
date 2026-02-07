from ignite import metrics
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
from ignite.contrib.handlers import global_step_from_engine
import torch


def create_trainer(
    model, criterion, device, optimizer, scheduler, train_loader, val_loader
):
    def loss_fn(preds, y):
        # Permute predictions to match CrossEntropyLoss expected input format (batch, dim, seq)
        preds = preds.permute(0, 2, 1)
        loss = criterion(preds, y)
        return loss

    def train(engine, batch):
        model.train()
        x, y = batch[0].to(device), batch[1].to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        return loss.item()

    def evaluate(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_preds = model(x)
        return y_preds, y

    def accuracy_transform(output):
        preds, y = output

        preds_flat = preds.reshape(-1, preds.size(-1))
        y_flat = y.reshape(-1)

        mask = y_flat != 0

        preds_masked = preds_flat[mask]
        y_masked = y_flat[mask]

        return preds_masked, y_masked

    model_metrics = {
        "accuracy": Accuracy(output_transform=accuracy_transform),
        "loss": Loss(loss_fn),
    }

    trainer = Engine(train)
    train_evaluator = Engine(evaluate)
    val_evaluator = Engine(evaluate)

    for name, metric in model_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in model_metrics.items():
        metric.attach(val_evaluator, name)

    def log_metrics(trainer):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

        train_metrics = train_evaluator.state.metrics
        val_metrics = val_evaluator.state.metrics

        print(
            f"Training Results - Epoch: {trainer.state.epoch} Avg accuracy: {train_metrics['accuracy']:.4f} Loss: {train_metrics['loss']:.4f}"
        )
        print(
            f"Validation Results - Epoch: {trainer.state.epoch} Avg accuracy: {val_metrics['accuracy']:.4f} Loss: {val_metrics['loss']:.4f}"
        )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=100),
        lambda engine: (
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        ),
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics)

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    def early_stopping_function(engine):
        return -engine.state.metrics["loss"]

    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=lambda *_: trainer.state.epoch,
    )

    latest_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=1,
        filename_prefix="latest",
        global_step_transform=lambda *_: trainer.state.epoch,
    )

    periodic_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=3,
        filename_prefix="periodic",
        global_step_transform=lambda *_: trainer.state.epoch,
    )

    early_stopping = EarlyStopping(
        patience=3,
        score_function=early_stopping_function,
        trainer=trainer,
        min_delta=0.001,
    )

    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"model": model}
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, latest_checkpoint, {"model": model}
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=5), periodic_checkpoint, {"model": model}
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    tb_logger = TensorboardLogger(log_dir="./tb-logs")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, loader, evaluator in [
        ("training", train_loader, train_evaluator),
        ("validation", val_loader, val_evaluator),
    ]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["accuracy", "loss"],
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )

    return trainer, tb_logger


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    num_warmup_steps = max(1, int(num_warmup_steps))
    num_training_steps = max(num_warmup_steps + 1, int(num_training_steps))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps
    )

    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_training_steps - num_warmup_steps), eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )

    return scheduler
