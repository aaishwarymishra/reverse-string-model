import yaml
import argparse
from dataset import ReverseStringDataset, create_dataloader
from model import ReverseStringModel
import trainer as trainer_utils
from ignite.contrib.handlers import TensorboardLogger
import torch
import torch.nn as nn
import importlib
from typing import Any


def _parse_methods(value: Any, key: str) -> Any:
    if isinstance(value, str):
        if value.startswith("lambda"):
            return eval(value, {"__builtins__": {}})
        else:
            func_path = value
            if not isinstance(func_path, str) or "." not in func_path:
                raise ValueError(
                    f"Invalid function path for '{key}': {func_path!r}")
            module_name, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
    raise ValueError(f"Invalid value for '{key}': {value!r}. Must be a string starting with 'lambda' or a module path.")


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a reverse string model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train the model on (overrides config).",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    # Override device if provided
    if args.device:
        config["trainer"]["device"] = args.device

    device = config["trainer"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Create datasets
    dataset_config = config["dataset"]
    dataset = ReverseStringDataset(
        min=dataset_config.get("min", 5),
        max=dataset_config.get("max", 100),
        size=dataset_config.get("size", 10000),
        fixed_length=dataset_config.get("fixed_length"),
    )

    train_loader, val_loader = create_dataloader(
        dataset=dataset,
        batch_size=dataset_config.get("batch_size", 64),
        shuffle=dataset_config.get("shuffle", True),
        train_split=dataset_config.get("train_split", 0.8),
    )

    # Create model
    model_config = config.get("model", {})
    model = ReverseStringModel(
        num_layers=model_config.get("num_layers", 2),
        embed_dim=model_config.get("embed_dim", 128),
        intermediate=model_config.get("intermediate", 512),
        heads=model_config.get("heads", 4),
        vocab_size=len(dataset.char_to_idx),
        pad_idx=dataset.char_to_idx.get("<PAD>"),
    ).to(device)

    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training components
    pad_idx = dataset.char_to_idx.get("<PAD>")
    if "loss_fn" in config:
        criterion = _parse_methods(config["loss_fn"], "loss_fn")
    elif "loss" not in config:
        raise ValueError("Loss configuration is missing in the config file.")
    else:
        criterion = trainer_utils.get_loss(config.get("loss", {}), pad_idx=pad_idx)
    optimizer = trainer_utils.get_optimizer(
        config.get("optimizer", {}), model.parameters())

    trainer = trainer_utils.create_trainer_from_config(
        model, criterion, optimizer, device, config.get("trainer", {})
    )

    # Attach scheduler if configured
    if "scheduler" in config:
        total_steps = len(train_loader) * \
            config.get("trainer", {}).get("epochs", 1)
        scheduler, scheduler_handler, scheduler_event = trainer_utils.create_scheduler_from_config(
            config["scheduler"], optimizer, total_steps
        )
        if scheduler and scheduler_handler:
            trainer.add_event_handler(scheduler_event, scheduler_handler)

    # Setup evaluation and metrics
    if "metrics_fn" in config:
        metrics = _parse_methods(config["metrics_fn"], "metrics_fn")()
    else:
        metrics = trainer_utils.get_metrics_from_config(
            config.get("metrics", []), criterion)
    evaluators = {}
    if metrics:
        evaluators["train_evaluator"] = trainer_utils.create_evaluator_from_config(
            model, criterion, metrics, device, config.get("evaluator", {})
        )
        evaluators["val_evaluator"] = trainer_utils.create_evaluator_from_config(
            model, criterion, metrics, device, config.get("evaluator", {})
        )

    # Attach handlers (checkpoints, early stopping, etc.)
    context = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler if "scheduler" in config else None,
        "train_evaluator": evaluators.get("train_evaluator"),
        "val_evaluator": evaluators.get("val_evaluator"),
    }
    if "handler_fn" in config:
        handlers = _parse_methods(config["handler_fn"], "handler_fn")(trainer, context)
    else:
        handlers = trainer_utils.create_handlers_from_config(
            config.get("handlers", []), trainer, context
        )
    # Setup logging
    trainer_utils.log_metrics(
        trainer,
        train_evaluator=evaluators.get("train_evaluator"),
        val_evaluator=evaluators.get("val_evaluator"),
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Start training
    trainer.run(train_loader, max_epochs=config.get(
        "trainer", {}).get("epochs", 10))

    # Clean up handlers
    for handler in handlers.values():
        if isinstance(handler, TensorboardLogger):
            handler.close()


if __name__ == "__main__":
    main()
