import sys

sys.path.append("../")

from utils import preprocess_bank
import torch
import torch.nn as nn
from model import BankModel
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import copy
import os

seeds = [
    3279724221,
    2863102059,
    1930825890,
    2841416373,
    586420623,
    123,
    580586463,
    42,
    610164470,
    2610654169,
]

for seed in seeds:

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = "mms"
    p = 0.2
    X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = preprocess_bank(seed)
    train_len = len(X_train)
    val_len = len(X_val)
    test_len = len(X_test)

    model = BankModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()

    epochs = 500

    batch_size = 16

    train_loss, test_loss = 0, 0

    class_weights = [1 / (y_train == i).sum() for i in range(2)]
    sample_weights = [class_weights[int(i)] for i in y_train]

    train_dataset = TensorDataset(X_train, y_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    best_model = None
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for x_i, y_i in train_loader:
            optimizer.zero_grad()

            y_logits = model(x_i).squeeze(1)
            loss = criterion(y_logits, y_i)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze()
            val_loss = criterion(val_logits, y_val)
            val_pred = torch.round(torch.sigmoid(val_logits))
            val_f1 = f1_score(
                y_val.detach().numpy(), val_pred.detach().numpy(), zero_division="warn"
            )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            train_loss = round(loss.item(), 4)
            val_loss_val = round(val_loss.item(), 4)
            print(f"Epoch: {epoch + 1} | Loss: {train_loss} | Val Loss: {val_loss_val}")

    if not os.path.exists("../saved_models/bank/"):
        os.makedirs("../saved_models/bank")

    torch.save(
        best_model,
        "../saved_models/bank/bank-{}-{}-{}-{}.pt".format(
            round(best_val_f1, 3), seed, scaler, p
        ),
    )
