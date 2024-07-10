import sys

sys.path.append("../")

import torch
from model import BankModel
from utils import equal_opp_difference, preprocess_bank
from sa import SimulatedAnnealingRepair
import os


if __name__ == "__main__":

    if not os.path.exists("./sa_runs_1hr"):
        os.makedirs("./sa_runs_1hr")

    layer_sizes = [32, 32]
    k_min = 2
    k_max = 24
    model_paths = [
        "bank-0.548-3279724221-mms-0.2.pt",
        "bank-0.554-2863102059-mms-0.2.pt",
        "bank-0.552-1930825890-mms-0.2.pt",
        "bank-0.547-2841416373-mms-0.2.pt",
        "bank-0.546-586420623-mms-0.2.pt",
        "bank-0.563-123-mms-0.2.pt",
        "bank-0.554-580586463-mms-0.2.pt",
        "bank-0.541-42-mms-0.2.pt",
        "bank-0.559-610164470-mms-0.2.pt",
        "bank-0.565-2610654169-mms-0.2.pt",
    ]

    # values of the 10 age groups
    sens_classes_all = [
        0.0,
        0.11111111,
        0.22222222,
        0.33333333,
        0.44444444,
        0.55555556,
        0.66666667,
        0.77777778,
        0.88888889,
        1.0,
    ]
    # choose first and fourth age groups
    sens_classes = [sens_classes_all[0], sens_classes_all[3]]

    for model_path in model_paths:
        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        seed = int(seed)
        dropout = float(dropout)
        model = BankModel(p=dropout)
        model.load_state_dict(torch.load("../saved_models/bank/{}".format(model_path)))
        model.eval()

        _, _, X_val, _, _, y_val, sens_idx = preprocess_bank(seed)
        X_val = X_val
        y_val = y_val
        sens_val = X_val[:, sens_idx]

        y_pred = (torch.sigmoid(model(X_val)).view(-1) > 0.5).float()
        baseline_eod = equal_opp_difference(y_val, y_pred, sens_val)

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            sens_classes=sens_classes,
            logfile="./sa_runs_1hr/bank_sa_{}.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")
