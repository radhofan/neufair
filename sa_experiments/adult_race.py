import sys

sys.path.append("../")

import torch
from model import AdultCensusModel
from utils import equal_opp_difference, preprocess_adult_census
from sa import SimulatedAnnealingRepair
import os


if __name__ == "__main__":

    if not os.path.exists("./sa_runs_1hr"):
        os.makedirs("./sa_runs_1hr")

    layer_sizes = [64, 128, 64]
    k_min = 2
    k_max = 50
    sens_classes = [0.0, 1.0]

    model_paths = [
        "adult-0.6732-839941478-ss-0.1.pt",
        "adult-0.6861-1207102389-ss-0.1.pt",
        "adult-0.6833-449024256-ss-0.1.pt",
        "adult-0.6844-1926335047-ss-0.1.pt",
        "adult-0.6872-3409747132-ss-0.1.pt",
        "adult-0.6777-362233404-ss-0.1.pt",
        "adult-0.6774-2837556887-ss-0.1.pt",
        "adult-0.684-42-ss-0.1.pt",
        "adult-0.666-973920294-ss-0.1.pt",
        "adult-0.676-123-ss-0.1.pt",
    ]

    adult_race = (23, 27)

    for model_path in model_paths:
        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        seed = int(seed)
        dropout = float(dropout)
        model = AdultCensusModel(p=dropout)
        model.load_state_dict(torch.load("../saved_models/adult/{}".format(model_path)))
        model.eval()

        _, _, X_val, _, _, y_val, sens_idx = preprocess_adult_census(seed)
        X_val = X_val
        y_val = y_val
        sens_val = X_val[:, adult_race[0] : adult_race[1]]

        y_pred = (torch.sigmoid(model(X_val)).view(-1) > 0.5).float()
        baseline_eod = equal_opp_difference(
            y_val, y_pred, sens_val, dataset="adult_race"
        )

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            sens_idx_range=adult_race,
            sens_multi_dataset="adult_race",
            logfile="./sa_runs_1hr/adult_race_sa_{}.log".format(seed),
        )

        obj.estimate_init_temp_and_run(chi_0=0.75, T0=5.0, p=5, eps=1e-3, decay="log")
