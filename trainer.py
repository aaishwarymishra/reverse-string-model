from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite import engine
from ignite import metrics as ignite_metrics
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
import torch
import torch.optim.lr_scheduler as lr_schedulers
import ignite.contrib.handlers.tensorboard_logger as tb_contrib
from typing import Dict, Any, Tuple, Callable
import importlib


def get_dataloader(dataset, batch_size: int = 64, shuffle: bool = True, train_split=None) -> tuple[torch.utils.data.DataLoader,]:
    """Create dataloader from dataset with specified batch size and shuffle."""
    if train_split is not None:
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        return (
            torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle),
            torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False),
        )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), None


def get_loss(loss_config: dict, pad_idx: int = None) -> torch.nn.Module:
    """Instantiate loss function from config."""
    loss_type = loss_config.get("type", "CrossEntropyLoss")
    loss_args = loss_config.get("args", {})

    if pad_idx is not None and loss_type == "CrossEntropyLoss":
        loss_args["ignore_index"] = pad_idx

    loss_cls = getattr(torch.nn, loss_type, None)
    if loss_cls is None:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss_cls(**loss_args)


def get_optimizer(optimizer_config: dict, model_params) -> torch.optim.Optimizer:
    """Instantiate optimizer from config."""
    optimizer_type = optimizer_config.get("type", "Adam")
    optimizer_args = optimizer_config.get("args", {})

    optimizer_cls = getattr(torch.optim, optimizer_type, None)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer_cls(model_params, **optimizer_args)


def create_scheduler_from_config(
    scheduler_config: Dict[str, Any], optimizer: torch.optim.Optimizer, total_iters: int
) -> Tuple[torch.optim.lr_scheduler.LRScheduler, Callable]:
    """
    Instantiate scheduler from config and return the scheduler and event function.
    For Ignite, we use StepLR scheduler attached via create_ignite_lr_scheduler_handler.
    """
    if not scheduler_config:
        return None, None

    scheduler_type = scheduler_config.get("type", "CosineAnnealingLR")
    scheduler_args = scheduler_config.get("args", {}).copy()
    event_type = scheduler_config.get("event", "ITERATION_COMPLETED")
    scheduler_handler_args = scheduler_config.get("handler_args", {})

    scheduler = getattr(lr_schedulers, scheduler_type)(
        optimizer, **scheduler_args)
    scheduler_handler = LRScheduler(scheduler, **scheduler_handler_args)

    return scheduler, scheduler_handler, event_type


def create_trainer_from_config(
    model: torch.nn.Module,
    criterion: torch.nn.Module | Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: Dict[str, Any],
):
    """
    Create trainer, evaluators, scheduler, and handlers from YAML config.
    Returns (trainer, val_evaluator, handlers_dict) or (trainer, None, None) if minimal setup.

    Args:
        model: The model to train.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to train on ('cpu' or 'cuda').
        config: YAML config dict with trainer, metrics, scheduler, handlers sections.
        train_loader: Training data loader (optional, for evaluators).
        val_loader: Validation data loader (optional, for evaluators).

    Returns:
        Tuple of (trainer, val_evaluator, handlers_dict) if full config,
        or (trainer, None, None) if minimal config.
    """
    # Create trainer
    trainer_type = config.get("trainer", {}).get(
        "type", "create_supervised_trainer")
    trainer_args = config.get("trainer", {}).get("args", {})
    trainer = getattr(engine, trainer_type)(model, optimizer,
                                            criterion, device=device, **trainer_args)
    return trainer

    # Early return if no evaluators/handlers configured


def create_evaluator_from_config(
    model: torch.nn.Module,
    criterion: torch.nn.Module | Callable,
    metrics: Dict[str, Any],
    device: str,
    config: Dict[str, Any],
) -> engine.Engine:
    """
    Create evaluator from YAML config.
    Uses Ignite's built-in create_supervised_evaluator.
    """
    evaluator_type = config.get("type", "create_supervised_evaluator")
    evaluator_args = config.get("args", {})

    evaluator = getattr(engine, evaluator_type)(
        model, metrics=metrics, device=device, **evaluator_args)

    return evaluator


def get_metrics_from_config(metrics_config: list[dict[str, Any]], criterion: torch.nn.Module | Callable) -> Dict[str, Any]:
    """
    Instantiate metrics from config.
    Currently supports Accuracy and Loss.
    """
    metrics = {"loss": ignite_metrics.Loss(criterion)}
    for metric_config in metrics_config:
        metric_type = metric_config.get("type", "Accuracy")
        metric_name = metric_config.get("name", metric_type)
        metric_args = metric_config.get("args", {}).copy()

        for key, value in metric_args.items():
            metric_args[key] = _parse_config_value(
                value, f"metric.{metric_name}.{key}")
        metrics[metric_name] = getattr(
            ignite_metrics, metric_type)(**metric_args)

    return metrics


def _resolve_event(event_str: str, engine: engine.Engine = None) -> Any:
    """
    Resolve event string like 'EPOCH_COMPLETED' or 'EPOCH_COMPLETED(every=5)' to Events constant.
    """
    if "(" in event_str:
        # Handle Events.EPOCH_COMPLETED(every=5) syntax
        match = event_str.split("(")
        event_name = match[0]
        args_str = "(" + match[1]
        event_base = getattr(Events, event_name)
        # eval the event call with restricted scope
        return eval(f"event_base{args_str}", {"__builtins__": {}, "event_base": event_base}, {})
    else:
        return getattr(Events, event_str)


def _parse_config_value(value: Any, key: str, context: Dict[str, Any] = None) -> Any:
    """
    Parse a configuration value which could be:
    1. A lambda string: "lambda x: ..."
    2. A function mapping: {"function": "module.fn"}
    3. An object reference: "model" (checked against context)
    4. A raw value
    """
    if isinstance(value, str):
        if value.startswith("lambda"):
            # evaluate lambda in restricted globals to reduce risk
            return eval(value, {"__builtins__": {}}, {})
        if context and value in context:
            return context[value]
    elif isinstance(value, dict) and "function" in value:
        func_path = value["function"]
        if not isinstance(func_path, str) or "." not in func_path:
            raise ValueError(
                f"Invalid function path for '{key}': {func_path!r}")
        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    return value


def get_handlers_from_config(
    handlers_config: list[dict[str, Any]],
    trainer: engine.Engine,
    evaluators: Dict[str, engine.Engine],
    model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Instantiate and attach handlers from config.
    Currently supports ModelCheckpoint, EarlyStopping, and TensorboardLogger.

    Args:
        handlers_config: List of handler configurations from YAML.
        trainer: The training engine.
        evaluators: Dict of evaluators keyed by name (e.g., {"val_evaluator": ..., "train_evaluator": ...}).
        model: The model instance.

    Returns:
        Dictionary of instantiated handlers.
    """
    if not handlers_config:
        return {}

    handlers = {}
    context = {
        "trainer": trainer,
        "model": model,
        **evaluators,
    }

    for handler_config in handlers_config:
        handler_type = handler_config.get("type", "Checkpoint")
        handler_args = handler_config.get("args", {}).copy()
        event_str = handler_config.get("event", "EPOCH_COMPLETED")
        targets = handler_config.get("targets", [])

        # Resolve object references, lambdas, and functions in args
        for key, value in handler_args.items():
            if isinstance(value, dict) and "to_save" in value:
                handler_args[key] = {
                    k: _parse_config_value(v, f"{key}.{k}", context)
                    for k, v in value.items()
                }
            else:
                handler_args[key] = _parse_config_value(value, key, context)

        # Instantiate handler based on type
        handler_name = handler_config.get(
            "name", f"{handler_type}_{len(handlers)}")

        if handler_type == "Checkpoint":
            handler = ModelCheckpoint(**handler_args)
            handlers[handler_name] = handler

            # Attach to specified targets
            for target_name in targets:
                if target_name not in context:
                    raise ValueError(
                        f"Target '{target_name}' not found in context")
                target_engine = context[target_name]
                event = _resolve_event(event_str)

                # Extract to_save from handler_args or reconstruct
                to_save_dict = handler_args.get("to_save", {"model": model})
                target_engine.add_event_handler(event, handler, to_save_dict)

        elif handler_type == "EarlyStopping":
            handler = EarlyStopping(**handler_args)
            handlers[handler_name] = handler

            # Attach to specified targets
            for target_name in targets:
                if target_name not in context:
                    raise ValueError(
                        f"Target '{target_name}' not found in context")
                target_engine = context[target_name]
                event = _resolve_event(event_str)
                target_engine.add_event_handler(event, handler)

        elif handler_type == "TensorboardLogger":
            tb_logger = TensorboardLogger(**handler_args)
            handlers[handler_name] = tb_logger

            # Attach multiple output handlers based on attachments list
            attachments = handler_config.get("attachments", [])
            for attachment in attachments:
                target_name = attachment.get("target", "trainer")
                if target_name not in context:
                    raise ValueError(
                        f"Target '{target_name}' not found in context")
                target_engine = context[target_name]
                event_str = attachment.get("event", "ITERATION_COMPLETED")
                event = _resolve_event(event_str)

                # Extract and parse attachment fields
                tag = attachment.get("tag", target_name)
                metric_names = attachment.get("metric_names")

                # Resolve lambdas/functions for transformations
                output_transform = _parse_config_value(
                    attachment.get("output_transform"), "output_transform")
                global_step_transform = _parse_config_value(
                    attachment.get("global_step_transform"), "global_step_transform")

                if output_transform or metric_names:
                    tb_logger.attach_output_handler(
                        target_engine,
                        event_name=event,
                        tag=tag,
                        output_transform=output_transform,
                        metric_names=metric_names,
                        global_step_transform=global_step_transform,
                    )

            # Process generic log handlers (e.g. OptimizerParamsHandler)
            log_handlers_cfg = handler_config.get("log_handlers", [])
            for lh_cfg in log_handlers_cfg:
                target_name = lh_cfg.get("target", "trainer")
                target_engine = context[target_name]
                event = _resolve_event(lh_cfg.get(
                    "event", "ITERATION_STARTED"))

                lh_type = lh_cfg.get("handler_type")
                lh_args = {
                    k: _parse_config_value(v, f"log_handler_args.{k}", context)
                    for k, v in lh_cfg.get("args", {}).items()
                }

                handler_cls = getattr(tb_contrib, lh_type)
                log_handler = handler_cls(**lh_args)

                tb_logger.attach(
                    target_engine,
                    log_handler=log_handler,
                    event_name=event
                )

    return handlers


def log_metrics(trainer: engine.Engine, train_evaluator=None, val_evaluator=None, train_loader=None, val_loader=None):
    def log(engine: engine.Engine):
        if train_evaluator is not None and train_loader is not None:
            train_evaluator.run(train_loader)
            train_metrics = train_evaluator.state.metrics
        if val_evaluator is not None and val_loader is not None:
            val_evaluator.run(val_loader)
            val_metrics = val_evaluator.state.metrics
        print(f"Epoch {trainer.state.epoch} - Train Metrics: {train_metrics if train_evaluator else 'N/A'}, Val Metrics: {val_metrics if val_evaluator else 'N/A'}")

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log)
