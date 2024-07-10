import sys

sys.path.append("../")

from utils import preprocess_adult_census
import torch
import torch.nn as nn
from model import AdultCensusModel
from sklearn.metrics import f1_score
import numpy as np
import copy
import os


seeds = [
    839941478,
    1207102389,
    449024256,
    1926335047,
    3409747132,
    362233404,
    2837556887,
    42,
    973920294,
    123,
]

for seed in seeds:

    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = preprocess_adult_census(
        seed
    )

    model = AdultCensusModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()

    epochs = 20000
    batch_size = X_train.shape[0]

    train_loss, test_loss = 0, 0

    best_val_f1 = 0
    best_val_model = None

    for epoch in range(epochs):
        if epoch == 10000:
            for g in optimizer.param_groups:
                g["lr"] = 0.001

        ### Training
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_logits = model(X_train).squeeze(1)
        loss = criterion(y_logits, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        ### Testing
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
                best_val_model = copy.deepcopy(model)

        if (epoch + 1) % 1000 == 0:
            train_loss = round(loss.item(), 4)
            val_loss_val = round(val_loss.item(), 4)
            print(
                f"Epoch: {epoch + 1} | Loss: {train_loss} | Val Loss: {val_loss_val} | Best Val F1: {best_val_f1} | Current Val F1: {val_f1}"
            )

    if not os.path.exists("../saved_models/adult/"):
        os.makedirs("../saved_models/adult")

    torch.save(
        best_val_model.state_dict(),
        "../saved_models/adult/adult-{}-{}-{}-{}.pt".format(
            round(best_val_f1, 4), seed, "ss", 0.1
        ),
    )
