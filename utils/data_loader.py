import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MHSMA(Dataset):
    def __init__(
        self, X_filename, dir, Y_filename, just_normal, normal_class, train, transform
    ):
        self.transform = transform
        file = os.path.join(dir, X_filename)
        self.data = np.load(file)
        self.data = self.data / 255.0
        self.data = self.data.reshape(-1, 1, 64, 64)
        self.data = torch.from_numpy(self.data).to(torch.float32)

        file = os.path.join(dir, Y_filename)
        self.targets = np.load(file)
        self.targets = self.targets.reshape(-1, 1)

        if just_normal and train:
            mask = np.where(self.targets == normal_class)[0]
            self.data = self.data[mask]
            self.targets = self.targets[mask]

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            X = self.transform(X)
        return X, y


def load_data(batch_size, just_normal, normal_class, augmentation, mode):
    train_transform = [transforms.Normalize(mean=0.5, std=0.04)]
    test_transform = [transforms.Normalize(mean=0.5, std=0.04)]
    if augmentation:
        train_transform.extend(
            [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=360),
            ]
        )

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    train_set = MHSMA(
        dir="/content/mhsma-dataset/mhsma/",
        X_filename="x_64_train.npy",
        Y_filename=f"y_{mode}_train.npy",
        just_normal=just_normal,
        normal_class=normal_class,
        train=True,
        transform=train_transform,
    )
    valid_set = MHSMA(
        dir="/content/mhsma-dataset/mhsma/",
        X_filename="x_64_valid.npy",
        Y_filename=f"y_{mode}_valid.npy",
        just_normal=False,
        normal_class=normal_class,
        train=False,
        transform=test_transform,
    )
    test_set = MHSMA(
        dir="/content/mhsma-dataset/mhsma/",
        X_filename="x_64_test.npy",
        Y_filename=f"y_{mode}_test.npy",
        just_normal=False,
        normal_class=normal_class,
        train=False,
        transform=test_transform,
    )

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader
