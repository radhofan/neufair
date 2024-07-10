import sys

sys.path.append("../")

import torch
from model import DefaultModel
from utils import equal_opp_difference, preprocess_default
from sa import SimulatedAnnealingRepair
import os


if __name__ == "__main__":

    if not os.path.exists("./random_runs_1hr"):
        os.makedirs("./random_runs_1hr")

    layer_sizes = [16, 16, 16]
    k_min = 2
    k_max = 20
    sens_classes = [0.0, 1.0]
    model_paths = [
        "default-0.5413-3774230515-mms-0.2.pt",
        "default-0.535-42-mms-0.2.pt",
        "default-0.543-123-mms-0.2.pt",
        "default-0.5307-1492128891-mms-0.2.pt",
        "default-0.5303-1113378588-mms-0.2.pt",
        "default-0.539-2602142440-mms-0.2.pt",
        "default-0.5544-1211744201-mms-0.2.pt",
        "default-0.5376-1210912848-mms-0.2.pt",
        "default-0.542-2672642440-mms-0.2.pt",
        "default-0.5405-2860010955-mms-0.2.pt",
        "default-0.5238-1959075396-mms-0.2.pt",
    ]

    for model_path in model_paths:
        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        seed = int(seed)
        dropout = float(dropout)
        model = DefaultModel(p=dropout)
        model.load_state_dict(
            torch.load("../saved_models/default/{}".format(model_path))
        )
        model.eval()

        _, _, X_val, _, _, y_val, sens_idx = preprocess_default(seed)
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
            logfile="./random_runs_1hr/default_sa_{}.log".format(seed),
        )

        obj.run_sa(random_walk=True)
