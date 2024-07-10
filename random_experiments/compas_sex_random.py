import sys

sys.path.append("../")

import torch
from model import CompasModel
from utils import equal_opp_difference, preprocess_compas
from sa import SimulatedAnnealingRepair
import os


if __name__ == "__main__":

    if not os.path.exists("./random_runs_1hr"):
        os.makedirs("./random_runs_1hr")

    layer_sizes = [32, 32]
    k_min = 2
    k_max = 24
    model_paths = [
        "compas-0.966-1207592030-mms-0.2.pt",
        "compas-0.97-3349636645-mms-0.2.pt",
        "compas-0.971-625199780-mms-0.2.pt",
        "compas-0.966-42-mms-0.2.pt",
        "compas-0.961-1437115122-mms-0.2.pt",
        "compas-0.969-3437336315-mms-0.2.pt",
        "compas-0.97-2320978921-mms-0.2.pt",
        "compas-0.969-123-mms-0.2.pt",
        "compas-0.968-3220064806-mms-0.2.pt",
        "compas-0.969-1118391417-mms-0.2.pt",
    ]

    attr = "sex"

    for model_path in model_paths:
        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        seed = int(seed)
        dropout = float(dropout)
        model = CompasModel(p=dropout)
        model.load_state_dict(
            torch.load("../saved_models/compas/{}".format(model_path))
        )
        model.eval()

        _, _, X_val, _, _, y_val, sens_idx = preprocess_compas(seed, attr)
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
            attr=attr,
            logfile="./random_runs_1hr/compas_sa_sex_{}.log".format(seed),
        )

        obj.run_sa(random_walk=True)
