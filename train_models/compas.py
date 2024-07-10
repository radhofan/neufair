import sys

sys.path.append("../")

from utils import preprocess_compas
import torch
import torch.nn as nn
from model import CompasModel
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
import os


seeds = [
    1207592030,
    3349636645,
    625199780,
    42,
    1437115122,
    3437336315,
    2320978921,
    123,
    3220064806,
    1118391417,
]


for seed in seeds:

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = "mms"
    p = 0.2
    X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = preprocess_compas(
        seed, scaler=scaler
    )

    model = CompasModel(p=p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCEWithLogitsLoss()

    epochs = 5000

    batch_size = 256

    train_loss, test_loss = 0, 0

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    if not os.path.exists("../saved_models/compas"):
        os.makedirs("../saved_models/compas")

    torch.save(
        best_model,
        "../saved_models/compas/compas-{}-{}-{}-{}.pt".format(
            round(best_val_f1, 3), seed, scaler, p
        ),
    )
