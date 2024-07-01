import torch
from torch.autograd import Variable
from models.vgg import Vgg16
from models.custom_vgg import make_arch
from utils.data_loader import load_data
from utils.loss import MseDirectionLoss
from utils.attacks import fgsm_attack
from config import *
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train():
    set_seed(RANDOM_SEED)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        batch_size=BATCH_SIZE,
        just_normal=JUST_NORMAL,
        normal_class=NORMAL_CLASS,
        augmentation=AUGMENTATION,
        mode=MODE,
    )

    vgg = Vgg16().to(DEVICE)
    model = make_arch(
        CFG, use_bias=USE_BIAS, batch_norm=BATCH_NORM, target_layer=TARGET_LAYER
    ).to(DEVICE)

    criterion = MseDirectionLoss(
        LAMBDA, CRIT, VGG_IMPORTANT_LAYERS, MODEL_IMPORTANT_LAYERS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            if FGSM_ATTACK_ENABLE:
                X = fgsm_attack(model, vgg, criterion, X, EPSILON)

            X = Variable(X).to(DEVICE)

            output_pred = model.forward(X)
            output_real = vgg(X)

            total_loss = criterion(output_pred, output_real)
            if SCHEDULER:
                scheduler.step(total_loss)

            epoch_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            # Perform validation or testing here
            pass


if __name__ == "__main__":
    train()
