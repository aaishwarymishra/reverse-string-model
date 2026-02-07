import random
import string
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


def generate_string(min: int = 5, max: int = 100, fixed_length: int | None = None):
    length = random.randint(min, max) if fixed_length is None else fixed_length
    characters = string.ascii_letters + string.digits
    s = "".join(random.choice(characters) for _ in range(length))
    return s


def generate_dataset(min=5, max=100, size=1000, fixed_length: int | None = None):
    dataset = []
    for _ in range(size):
        dataset.append(generate_string(min, max, fixed_length=fixed_length))

    return dataset


class ReverseStringDataset(Dataset):
    def __init__(self, min=5, max=100, size=1000, fixed_length=None) -> None:
        super().__init__()
        self.data = generate_dataset(
            min=min, max=max, size=size, fixed_length=fixed_length
        )
        self.characters = string.ascii_letters + string.digits
        self.idx_to_char = {
            i + 4: self.characters[i] for i in range(0, len(self.characters))
        }
        self.char_to_idx = {value: key for key, value in self.idx_to_char.items()}
        self.idx_to_char.update({0: "<PAD>", 1: "<SOS>", 2: "<SEP>", 3: "<EOS>"})
        self.char_to_idx.update({"<PAD>": 0, "<SOS>": 1, "<SEP>": 2, "<EOS>": 3})
        self.pad_idx = self.char_to_idx["<PAD>"]
        self.sos_idx = self.char_to_idx["<SOS>"]
        self.eos_idx = self.char_to_idx["<EOS>"]
        self.sep_idx = self.char_to_idx["<SEP>"]

    def __getitem__(self, index):
        s = self.data[index]
        encoded_s = self.simple_encode(s)

        context = [self.sos_idx] + encoded_s + [self.sep_idx]

        reversed_s = encoded_s[::-1] + [self.eos_idx]

        x = context + reversed_s[:-1]

        y = [self.pad_idx] * (len(context) - 1) + reversed_s

        return torch.tensor(x), torch.tensor(y)

    def get(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    # encoding without special tokens, used for testing and debugging
    def simple_encode(self, s):
        encoded_s = []
        for c in s:
            if c in self.char_to_idx:
                encoded_s.append(self.char_to_idx[c])
            else:
                raise ValueError(f"Character '{c}' not in vocabulary")
        return encoded_s

    # encoding with special tokens, used for training
    def encode(self, s):
        encoded_s = [self.sos_idx]
        for c in s:
            if c in self.char_to_idx:
                encoded_s.append(self.char_to_idx[c])
            else:
                raise ValueError(f"Character '{c}' not in vocabulary")
        encoded_s.append(self.sep_idx)
        return encoded_s

    # decoding token indices back to a string
    def decode(self, indices):
        s = []

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        for idx in indices:
            if idx in [self.pad_idx, self.sos_idx, self.eos_idx, self.sep_idx]:
                continue

            if idx in self.idx_to_char:
                s.append(self.idx_to_char[idx])

        return "".join(s)


def collator(batch):
    x = [data[0] for data in batch]
    y = [data[1] for data in batch]

    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)

    return x, y


def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator
    )
