from torch.utils import data
import argparse
import yaml

from dataset import ReverseStringDataset, create_dataloader
from trainer import BaseTrainer, parse_function_from_string
import torch


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
    dataset = parse_function_from_string(
        dataset_config.get("path", "dataset.ReverseStringDataset")
    )(**dataset_config.get("kargs", {}))

    train_loader, val_loader = create_dataloader(
        dataset=dataset,
        **dataset_config.get("dataloader_kargs", {"batch_size": 32, "shuffle": True}),
    )
    # Create model
    model_config = config.get("model", {})
    model_path = model_config.get("path", "model.ReverseStringModel")
    model = parse_function_from_string(model_path)(**model_config.get("kargs", {})).to(
        device
    )

    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = BaseTrainer(
        model=model,
        device=device,
        cfg=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Start training
    trainer.run()

    # Clean up handlers
    tb_logger = trainer.handlers.get("tensorboard")
    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    main()
