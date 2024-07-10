import sys

sys.path.append("../")

import torch
from model import MEPS16Model
from utils import equal_opp_difference, preprocess_meps16
from sa import SimulatedAnnealingRepair
import os


if __name__ == "__main__":

    if not os.path.exists("./random_runs_1hr"):
        os.makedirs("./random_runs_1hr")

    layer_sizes = [128, 128, 128]
    k_min = 2
    k_max = 135
    model_paths = [
        "meps16-0.517-36821596-mms-0.1.pt",
        "meps16-0.542-42-mms-0.1.pt",
        "meps16-0.562-398198020-mms-0.1.pt",
        "meps16-0.541-123-mms-0.2.pt",
        "meps16-0.544-2579600519-mms-0.1.pt",
        "meps16-0.544-42-mms-0.2.pt",
        "meps16-0.535-3569605705-mms-0.1.pt",
        "meps16-0.544-123-mms-0.1.pt",
        "meps16-0.54-608130088-mms-0.1.pt",
        "meps16-0.528-289553502-mms-0.1.pt",
        "meps16-0.563-525558053-mms-0.1.pt",
        "meps16-0.558-2123741781-mms-0.1.pt",
    ]

    sens_classes_dict = {
        42: [-0.7830757, 1.2770158],
        608130088: [-0.7901597, 1.265567],
        123: [-0.78272235, 1.2775922],
        2579600519: [-0.7892723, 1.2669898],
        3569605705: [-0.7914029, 1.2635789],
        289553502: [-0.7947826, 1.2582057],
        398198020: [-0.7834291, 1.2764397],
        525558053: [-0.7844898, 1.2747139],
        2123741781: [-0.7947826, 1.2582057],
        36821596: [-0.7898047, 1.2661358],
    }

    for model_path in model_paths:
        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        seed = int(seed)
        dropout = float(dropout)
        model = MEPS16Model(p=dropout)
        model.load_state_dict(
            torch.load("../saved_models/{}/{}".format(dataset, model_path))
        )
        model.eval()

        X_train, _, X_val, y_train, _, y_val, sens_idx = preprocess_meps16(seed)
        y_val = y_val
        sens_val = X_val[:, sens_idx]

        sens_classes = sens_classes_dict[seed]

        y_pred = (torch.sigmoid(model(X_val)).view(-1) > 0.5).float()
        baseline_eod = equal_opp_difference(
            y_val, y_pred, sens_val, sens_classes=sens_classes
        )
        print(baseline_eod)

        obj = SimulatedAnnealingRepair(
            model_path=model_path,
            layer_sizes=layer_sizes,
            k_min=k_min,
            k_max=k_max,
            baseline_eod=baseline_eod,
            sens_classes=sens_classes,
            logfile="./random_runs_1hr/meps16_sa_{}.log".format(seed),
        )

        obj.run_sa(random_walk=True)
