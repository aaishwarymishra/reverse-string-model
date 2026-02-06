import pydantic
import argparse
from dataset import ReverseStringDataset, create_dataloader
from model import ReverseStringModel
from trainer import create_trainer, get_scheduler
import torch
import torch.nn as nn


class Config(pydantic.BaseModel):
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    max_length: int = 100
    fixed_length: int = 100
    scheduler: bool = True
    num_layers: int = 2
    embed_dim: int = 128
    intermediate: int = 512
    heads: int = 4
    train_size: int = 10000
    val_size: int = 1000
    device: str = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reverse string model.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of the input strings.",
    )
    parser.add_argument(
        "--fixed_length",
        type=int,
        default=100,
        help="Fixed length for input strings (if specified).",
    )
    parser.add_argument(
        "--scheduler",
        type=bool,
        default=True,
        help="Whether to use a learning rate scheduler.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension for the model."
    )
    parser.add_argument(
        "--intermediate",
        type=int,
        default=512,
        help="Intermediate dimension for the feedforward network.",
    )
    parser.add_argument(
        "--heads", type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        "--train_size", type=int, default=10000, help="Number of training samples."
    )
    parser.add_argument(
        "--val_size", type=int, default=1000, help="Number of validation samples."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train the model on (e.g., 'cpu' or 'cuda').",
    )
    args = parser.parse_args()
    return args


def get_config():
    args = parse_args()
    config = Config(**vars(args))
    return config


def main():
    config = get_config()
    train_dataset = ReverseStringDataset(
        min=5,
        max=config.max_length,
        size=config.train_size,
        fixed_length=config.fixed_length,
    )
    val_dataset = ReverseStringDataset(
        min=5,
        max=config.max_length,
        size=config.val_size,
        fixed_length=config.fixed_length,
    )
    train_loader = create_dataloader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        config.device = "cpu"
    model = ReverseStringModel(
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        intermediate=config.intermediate,
        heads=config.heads,
        vocab_size=len(train_dataset.char_to_idx),
        pad_idx=train_dataset.char_to_idx.get("<pad>"),
    ).to(config.device)

    criterion = (
        nn.CrossEntropyLoss(ignore_index=train_dataset.char_to_idx.get("<pad>"))
        if train_dataset.char_to_idx.get("<pad>") is not None
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler:
        total_steps = config.num_epochs * len(train_loader)
        scheduler = get_scheduler(optimizer, total_steps*0.1, total_steps)
    else:
        scheduler = None
    trainer = create_trainer(
        model, criterion, config.device, optimizer, scheduler, train_loader, val_loader
    )
    trainer.run(train_loader, max_epochs=config.num_epochs)


if __name__ == "__main__":
    main()
