import yaml
import argparse
from dataset import ReverseStringDataset, create_dataloader
from model import ReverseStringModel
from trainer 
from ignite.contrib.handlers import TensorboardLogger
import torch
import torch.nn as nn


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a reverse string model.")
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
    model_config = config["model"]
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

    # Create loss function
    pad_idx = dataset.char_to_idx.get("<PAD>")
    criterion = trainer.get_loss(config["loss"], pad_idx=pad_idx)

    # Create optimizer
    optimizer = trainer.get_optimizer(config["optimizer"], model.parameters())
    scheduler = None
    if "scheduler" in config:
        scheduler,scheduler_handler,scheduler_event = trainer.get_scheduler(config["scheduler"], optimizer)

    trainer = trainer.create_trainer_from_config(model, optimizer, criterion, device, config.get("trainer", {}))

    if scheduler is not None:
        trainer.add_event_handler(scheduler_event, scheduler_handler)

    metrics = trainer.get_metrics_from_config(config.get("metrics", {}), criterion)

    evaluators = {}
    if train_loader is not None and metrics:
        evaluators["train_evaluator"] = trainer.create_evaluator_from_config(model, device, metrics, config.get("evaluator", {}))
    if val_loader is not None and metrics:
        evaluators["val_evaluator"] = trainer.create_evaluator_from_config(model, device, metrics, config.get("evaluator", {}))

    handlers = trainer.create_handlers_from_config(config.get("handlers", {}), trainer, evaluators,model)

    

if __name__ == "__main__":
    main()
