import torch
import numpy as np
import random
from typing import List, Tuple
from model import (
    AdultCensusModel,
    CompasModel,
    BankModel,
    DefaultModel,
    MEPS16Model,
)
from utils import (
    preprocess_adult_census,
    preprocess_compas,
    preprocess_bank,
    preprocess_default,
    preprocess_meps16,
    equal_opp_difference,
)
from sklearn.metrics import f1_score, accuracy_score
import math
import logging
import datetime
import time

models = {
    "adult": AdultCensusModel,
    "compas": CompasModel,
    "bank": BankModel,
    "default": DefaultModel,
    "meps16": MEPS16Model,
}

dataloaders = {
    "adult": preprocess_adult_census,
    "compas": preprocess_compas,
    "bank": preprocess_bank,
    "default": preprocess_default,
    "meps16": preprocess_meps16,
}


class SimulatedAnnealingRepair:
    def __init__(
        self,
        model_path: str,  # Name of the saved .pth file. Expected format: dataset-valacc-seed-scaler-dropout.pt
        layer_sizes: List[int], # List of integers representing the number of neurons in each layer
        k_min: int, # Minimum number of neurons that can be dropped
        k_max: int, # Maximum number of neurons that can be dropped
        baseline_eod: float, # Baseline EOD value
        f1_threshold: float = 0.98, # F1 threshold in cost function
        max_time: int = 60, # Maximum running time in minutes
        sens_idx_range: Tuple[int, int] = (-1, -1), # Range of indices for non-binary sensitive attributes
        sens_multi_dataset: str = "none", # Dataset for non-binary sensitive attributes
        attr: str = "race", # Sensitive attribute to use for datasets that have more than one possible sensitive attribute (E.g. Race and Sex in Compas)
        init_temp: float = 2.0, # Initial temperature estimate
        f1_penalty_multiplier: float = 3.0, # F1 penalty multiplier in cost function
        max_iter_temp_init: int = 25_000, # Maximum number of iterations for calculating temperature
        sens_classes: List[float] = [0, 1], # List of possible sensitive attribute values
        logfile: str = "default.log", # Log file name
        saved_models_path: str = "../saved_models", # Path to saved models
    ):

        dataset, valacc, seed, scaler, dropout = model_path[:-3].split("-")
        self.dataset = dataset
        self.seed = int(seed)
        self.scaler = scaler
        self.dropout = float(dropout)
        self.f1_threshold = float(valacc) * f1_threshold

        self.layer_sizes = layer_sizes
        self.state_size_bits = sum(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.sens_classes = sens_classes

        self.k_min = k_min
        self.k_max = k_max
        self.max_time = max_time
        self.max_iter_temp_init = max_iter_temp_init

        self.T = init_temp

        self.f1_penalty = baseline_eod
        self.f1_penalty *= f1_penalty_multiplier

        self.layer_size_prefix = [0]
        for i in range(len(layer_sizes)):
            self.layer_size_prefix.append(self.layer_size_prefix[-1] + layer_sizes[i])

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Load model
        self.model = models[self.dataset](p=self.dropout)
        self.model.load_state_dict(
            torch.load("{}/{}/{}".format(saved_models_path, self.dataset, model_path))
        )

        # Load data
        # Compas has two possible sensitive attributes and needs to be handled separatel
        if self.dataset == "compas":
            X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = dataloaders[
                self.dataset
            ](self.seed, attr, self.scaler)
        else:
            X_train, X_test, X_val, y_train, y_test, y_val, sens_idx = dataloaders[
                self.dataset
            ](self.seed, self.scaler)
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.sens_idx = sens_idx

        # Non-binary sensitive attributes that are one-hot encoded require a range of indices
        if sens_idx_range[0] != -1 and sens_idx_range[1] != -1:
            self.sens_val = self.X_val[:, sens_idx_range[0] : sens_idx_range[1]]
            self.sens_test = self.X_test[:, sens_idx_range[0] : sens_idx_range[1]]
            self.sens_train = self.X_train[:, sens_idx_range[0] : sens_idx_range[1]]
        # Binary sensitive attributes require only one index (0 or 1)
        else:
            self.sens_val = self.X_val[:, self.sens_idx]
            self.sens_test = self.X_test[:, self.sens_idx]
            self.sens_train = self.X_train[:, self.sens_idx]

        self.sens_multi_dataset = sens_multi_dataset

        self.best_state = None
        self.best_cost = None
        self.log_file_name = logfile

        self.model.eval()

    def run_sa(
        self,
        decay: str = "log",
        decay_rate: float = 0.00009,
        min_temp: float = 1e-4,
        random_walk: bool = False,
    ):
        self.log_file = open(self.log_file_name, "a")
        logging.info("Started Simulated Annealing procedure")
        if random_walk:
            logging.info("Random walk enabled")
        iter = 0
        # Get current state
        current_state = self.get_randombits()
        current_fairness, current_f1, current_acc = self.compute_fairness(current_state)
        current_cost = self.get_cost_step(current_fairness, current_f1)
        self.log_file.write("Run start at {}\n".format(datetime.datetime.now()))

        end_time = self.max_time * 60 + datetime.datetime.now().timestamp()
        while datetime.datetime.now().timestamp() <= end_time:
            # Get new temperature
            if decay == "log":
                T = self.logarithmic_decay(iter)
            elif decay == "exp":
                T = self.exponential_decay(
                    iter, decay_rate=decay_rate, min_temp=min_temp
                )
            elif decay == "geom":
                T = self.geometric_decay(iter)
            else:
                raise Exception("Unknown decay type {}".format(decay))

            new_state = self.new_state(current_state)
            new_fairness, new_f1, new_acc = self.compute_fairness(new_state)
            new_cost = self.get_cost_step(new_fairness, new_f1)

            if new_cost <= current_cost:
                transition_prob = 1.0
                current_cost = new_cost
                current_state = new_state
            else:
                if random_walk:
                    transition_prob = 1.0
                else:
                    transition_prob = math.exp(-(new_cost - current_cost) / T)
                if transition_prob >= random.random():
                    current_cost = new_cost
                    current_state = new_state

            if self.best_cost == None or self.best_cost >= current_cost:
                self.best_cost = current_cost
                self.best_state = current_state

            self.log_file.write(
                "Time: {}, End of iteration {}: Current Cost: {}, Current State: {}, Best Cost: {}, Best State: {}, Transition Prob: {}\n".format(
                    datetime.datetime.now(),
                    iter,
                    current_cost,
                    current_state,
                    self.best_cost,
                    self.best_state,
                    transition_prob,
                )
            )

            if (iter + 1) % 1000 == 0:
                print("Current Iter: ", iter)
                print("Transition Prob: ", transition_prob)
                print("Best Cost: ", self.best_cost)
                print("Best State: ", self.best_state)

            iter += 1

        eo_val, f1_score_val, accuracy_score_val = self.compute_fairness(
            self.best_state
        )
        eo_test, f1_score_test, accuracy_score_test = self.compute_fairness(
            self.best_state, "test"
        )
        eo_train, f1_score_train, accuracy_score_train = self.compute_fairness(
            self.best_state, "train"
        )

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                eo_val, f1_score_val, accuracy_score_val
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                eo_test, f1_score_test, accuracy_score_test
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                eo_train, f1_score_train, accuracy_score_train
            )
        )

        self.log_file.write("\n\n Rounded to 3 decimal places\n")

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(eo_val * 100, 3),
                round(f1_score_val, 3),
                round(accuracy_score_val, 3),
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(eo_test, 3),
                round(f1_score_test, 3),
                round(accuracy_score_test, 3),
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(eo_train, 3),
                round(f1_score_train, 3),
                round(accuracy_score_train, 3),
            )
        )

        self.log_file.write("\n\n Baseline (rounded to 3) \n\n")

        (
            val_f1,
            val_eo,
            val_acc,
            test_f1,
            test_eo,
            test_acc,
            train_f1,
            train_eo,
            train_acc,
        ) = self.baseline_fairness()

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(val_eo * 100, 3), round(val_f1, 3), round(val_acc, 3)
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(test_eo, 3), round(test_f1, 3), round(test_acc, 3)
            )
        )

        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(train_eo, 3), round(train_f1, 3), round(train_acc, 3)
            )
        )

        self.log_file.write("Run end at {}\n".format(datetime.datetime.now()))

    def drop_one(self):
        self.log_file = open(self.log_file_name, "a")
        logging.info("Dropping one at a time procedure")
        iter = 0
        # Get current state
        # current_state = self.get_randombits()
        current_state = 0
        current_fairness, current_f1, current_acc = self.compute_fairness(current_state)
        current_cost = self.get_cost_step(current_fairness, current_f1)
        self.log_file.write("Run start at {}\n".format(datetime.datetime.now()))
        self.best_state = current_state
        self.best_cost = current_cost

        # drop one neuron at a time
        for i in range(sum(self.layer_sizes)):

            new_state = current_state ^ (2**i)

            new_fairness, new_f1, new_acc = self.compute_fairness(new_state)
            new_cost = self.get_cost_step(new_fairness, new_f1)


            if self.best_cost >= new_cost:
                self.best_cost = new_cost
                self.best_state = new_state

            self.log_file.write(
                "Time: {}, End of iteration {}: Current Cost: {}, Current State: {}, Best Cost: {}, Best State: {}\n".format(
                    datetime.datetime.now(),
                    iter,
                    new_cost,
                    new_state,
                    self.best_cost,
                    self.best_state,
                )
            )

            iter += 1

        eo_val, f1_score_val, accuracy_score_val = self.compute_fairness(
            self.best_state
        )
        eo_test, f1_score_test, accuracy_score_test = self.compute_fairness(
            self.best_state, "test"
        )
        eo_train, f1_score_train, accuracy_score_train = self.compute_fairness(
            self.best_state, "train"
        )

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                eo_val, f1_score_val, accuracy_score_val
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                eo_test, f1_score_test, accuracy_score_test
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                eo_train, f1_score_train, accuracy_score_train
            )
        )

        self.log_file.write("\n\nRounded to 3 decimal places\n")

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(eo_val * 100, 3),
                round(f1_score_val, 3),
                round(accuracy_score_val, 3),
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(eo_test, 3),
                round(f1_score_test, 3),
                round(accuracy_score_test, 3),
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(eo_train, 3),
                round(f1_score_train, 3),
                round(accuracy_score_train, 3),
            )
        )

        self.log_file.write("\n\nBaseline (rounded to 3) \n\n")

        (
            val_f1,
            val_eo,
            val_acc,
            test_f1,
            test_eo,
            test_acc,
            train_f1,
            train_eo,
            train_acc,
        ) = self.baseline_fairness()

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(val_eo * 100, 3), round(val_f1, 3), round(val_acc, 3)
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(test_eo, 3), round(test_f1, 3), round(test_acc, 3)
            )
        )

        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(train_eo, 3), round(train_f1, 3), round(train_acc, 3)
            )
        )

        self.log_file.write("Run end at {}\n".format(datetime.datetime.now()))

    def brute_force(self):
        self.log_file = open(self.log_file_name, "a")
        logging.info("Brute force procedure. Generate all possible states.")
        iter = 0

        current_state = 0
        current_fairness, current_f1, current_acc = self.compute_fairness(current_state)
        current_cost = self.get_cost_step(current_fairness, current_f1)
        self.log_file.write("Run start at {}\n".format(datetime.datetime.now()))
        self.best_state = current_state
        self.best_cost = current_cost

        # Generate all 2**N states for a DNN
        for i in range(2 ** sum(self.layer_sizes)):

            new_state = i

            if self.k_min <= bin(new_state).count("1") <= self.k_max:
                continue

            new_fairness, new_f1, new_acc = self.compute_fairness(new_state)
            new_cost = self.get_cost_step(new_fairness, new_f1)

            if self.best_cost >= new_cost:
                self.best_cost = new_cost
                self.best_state = new_state

            self.log_file.write(
                "Time: {}, End of iteration {}: Current Cost: {}, Current State: {}, Best Cost: {}, Best State: {}\n".format(
                    datetime.datetime.now(),
                    iter,
                    new_cost,
                    new_state,
                    self.best_cost,
                    self.best_state,
                )
            )

            iter += 1

        eo_val, f1_score_val, accuracy_score_val = self.compute_fairness(
            self.best_state
        )
        eo_test, f1_score_test, accuracy_score_test = self.compute_fairness(
            self.best_state, "test"
        )
        eo_train, f1_score_train, accuracy_score_train = self.compute_fairness(
            self.best_state, "train"
        )

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                eo_val, f1_score_val, accuracy_score_val
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                eo_test, f1_score_test, accuracy_score_test
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                eo_train, f1_score_train, accuracy_score_train
            )
        )

        self.log_file.write("\n\nRounded to 3 decimal places\n")

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(eo_val * 100, 3),
                round(f1_score_val, 3),
                round(accuracy_score_val, 3),
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(eo_test, 3),
                round(f1_score_test, 3),
                round(accuracy_score_test, 3),
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(eo_train, 3),
                round(f1_score_train, 3),
                round(accuracy_score_train, 3),
            )
        )

        self.log_file.write("\n\nBaseline (rounded to 3) \n\n")

        (
            val_f1,
            val_eo,
            val_acc,
            test_f1,
            test_eo,
            test_acc,
            train_f1,
            train_eo,
            train_acc,
        ) = self.baseline_fairness()

        self.log_file.write(
            "\n\nVal EOD: {}, Val F1: {}, Val Acc: {}\n".format(
                round(val_eo * 100, 3), round(val_f1, 3), round(val_acc, 3)
            )
        )
        self.log_file.write(
            "Test EOD: {}, Test F1: {}, Test Acc: {}\n".format(
                round(test_eo, 3), round(test_f1, 3), round(test_acc, 3)
            )
        )
        self.log_file.write(
            "Train EOD: {}, Train F1: {}, Train Acc: {}\n\n".format(
                round(train_eo, 3), round(train_f1, 3), round(train_acc, 3)
            )
        )

        self.log_file.write("Run end at {}\n".format(datetime.datetime.now()))

    def estimate_initial_temp(
        self,
        chi_0: float = 0.8,
        T0: float = 5.0,
        p: int = 5,
        eps: float = 1e-3,
        random_walk: bool = True,
    ):
        iter = 0
        # Get current state
        current_state = self.get_randombits()
        current_fairness, current_f1, current_acc = self.compute_fairness(current_state)
        current_cost = self.get_cost_step(current_fairness, current_f1)
        states = []

        while iter < self.max_iter_temp_init:
            new_state = self.new_state(current_state)
            new_fairness, new_f1, new_acc = self.compute_fairness(new_state)
            new_cost = self.get_cost_step(new_fairness, new_f1)

            if new_cost <= current_cost:
                current_cost = new_cost
                current_state = new_state
            else:
                states.append((new_cost, current_cost))
                transition_prob = 1.0  # math.exp(-(new_cost - current_cost) / T)
                if transition_prob >= random.random():
                    current_cost = new_cost
                    current_state = new_state
                iter += 1

            if self.best_cost == None or self.best_cost >= current_cost:
                self.best_cost = current_cost
                self.best_state = current_state

            # Randomly re-sample new state for next iteration if not random walk
            if not random_walk:
                current_state = self.get_randombits()


        def compute_chi_t(states, temp):
            chi_t_num = 0
            chi_t_den = 0
            for state in states:
                chi_t_num += math.exp(-state[0] / temp)
                chi_t_den += math.exp(-state[1] / temp)
            return chi_t_num / chi_t_den

        T = T0
        while True:
            chi_t = compute_chi_t(states, T)
            if abs(chi_t - chi_0) <= eps:
                break
            T = T * (math.log(chi_t) / math.log(chi_0)) ** (1 / p)

        return T

    def estimate_init_temp_and_run(
        self,
        chi_0: float = 0.8,
        T0: float = 5.0,
        p: int = 5,
        eps: float = 1e-3,
        random_walk: bool = True,
        decay: str = "log",
        decay_rate: float = 0.00009,
        min_temp: float = 1e-4,
    ):
        T = self.estimate_initial_temp(
            chi_0=chi_0, T0=T0, p=p, eps=eps, random_walk=random_walk
        )
        self.T = T
        self.run_sa(decay=decay, decay_rate=decay_rate, min_temp=min_temp)

    def get_randombits(self):
        num = 0
        bits = random.randint(self.k_min, self.k_max)
        for bit in random.sample(range(self.state_size_bits), bits):
            num |= 1 << bit
        return num

    def new_state(self, current_state):
        # Generate new state
        def new_state_helper(current_state, state_size_bits):
            bit = random.randint(0, state_size_bits - 1)
            return current_state ^ (1 << bit)

        # Check for valid state
        while True:
            new_state = new_state_helper(current_state, self.state_size_bits)
            if self.k_min <= bin(new_state).count("1") <= self.k_max:
                break

        return new_state

    def get_cost_step(self, fairness, f1_score):
        return fairness + self.f1_penalty * (f1_score < self.f1_threshold)

    def linear_cooling_schedule(self):
        self.T *= 0.99

    def logarithmic_decay(self, iter):
        return self.T / math.log(2 + iter)

    def exponential_decay(
        self, iter, decay_rate: float = 0.005, min_temp: float = 1e-4
    ):
        return max(self.T * math.exp(-decay_rate * iter), min_temp)

    def geometric_decay(self, iter, decay_rate: float = 0.99, min_temp: float = 1e-4):
        return max(self.T * math.pow(decay_rate, iter), min_temp)

    def compute_fairness(self, state: int, dataset: str = "val") -> Tuple[float, float]:

        # Both dictionaries are 1 indexed by key and 0 indexed by value i.e first neuron of a first layer is 1, 0
        # Keys are layers and values are tuples (neuron_idx, neuron_val)
        old_neuron_values = {}
        # Keys are layers and values are neuron_idx
        neurons_to_drop = {}

        for i in range(1, len(self.layer_sizes) + 1):
            old_neuron_values[i] = []
            neurons_to_drop[i] = []

        # Identify which neurons to drop
        curr_layer = 0
        for idx, bit in enumerate(format(state, "0{}b".format(self.state_size_bits))):
            if idx == self.layer_size_prefix[curr_layer + 1]:
                curr_layer += 1
            if bit == "1":
                neurons_to_drop[curr_layer + 1].append(
                    idx - self.layer_size_prefix[curr_layer]
                )

        # Drop neurons and store old values
        for i in range(1, len(self.layer_sizes) + 1):
            layer = "layer{}".format(i)
            for j in neurons_to_drop[i]:
                old_neuron_values[i].append(
                    (j, getattr(self.model, layer).weight.data[j].clone())
                )
                getattr(self.model, layer).weight.data[j] = torch.zeros_like(
                    getattr(self.model, layer).weight.data[j]
                )

        X = self.X_val
        y = self.y_val
        sens = self.sens_val
        if dataset == "train":
            X = self.X_train
            y = self.y_train
            sens = self.sens_train
        elif dataset == "test":
            X = self.X_test
            y = self.y_test
            sens = self.sens_test

        # Compute fairness and f1_score
        y_pred = (torch.sigmoid(self.model(X)).view(-1) > 0.5).float()
        eo_val = equal_opp_difference(
            y,
            y_pred,
            sens,
            sens_classes=self.sens_classes,
            dataset=self.sens_multi_dataset,
        )
        f1_score_val = f1_score(y, y_pred)
        accuracy_score_val = accuracy_score(y, y_pred)

        # Reset dropped neurons
        for i in range(1, len(self.layer_sizes) + 1):
            layer = "layer{}".format(i)
            for j, val in old_neuron_values[i]:
                getattr(self.model, layer).weight.data[j] = val

        return eo_val, f1_score_val, accuracy_score_val

    def baseline_fairness(self):
        y_pred = (torch.sigmoid(self.model(self.X_val)).view(-1) > 0.5).float()
        eo_val = equal_opp_difference(
            self.y_val,
            y_pred,
            self.sens_val,
            sens_classes=self.sens_classes,
            dataset=self.sens_multi_dataset,
        )
        f1_score_val = f1_score(self.y_val, y_pred)
        accuracy_score_val = accuracy_score(self.y_val, y_pred)

        print("Val F1: ", round(f1_score_val, 3))
        print("Val EO: ", round(eo_val, 5) * 100)
        print("Val Accuracy: ", round(accuracy_score_val, 3))
        print()

        val_f1 = f1_score_val
        val_eo = eo_val
        val_acc = accuracy_score_val

        y_pred = (torch.sigmoid(self.model(self.X_test)).view(-1) > 0.5).float()
        eo_val = equal_opp_difference(
            self.y_test,
            y_pred,
            self.sens_test,
            sens_classes=self.sens_classes,
            dataset=self.sens_multi_dataset,
        )
        f1_score_val = f1_score(self.y_test, y_pred)
        accuracy_score_val = accuracy_score(self.y_test, y_pred)

        print("Test F1: ", round(f1_score_val, 3))
        print("Test EO: ", round(eo_val, 5) * 100)
        print("Test Accuracy: ", round(accuracy_score_val, 3))
        print()

        test_f1 = f1_score_val
        test_eo = eo_val
        test_acc = accuracy_score_val

        y_pred = (torch.sigmoid(self.model(self.X_train)).view(-1) > 0.5).float()
        eo_val = equal_opp_difference(
            self.y_train,
            y_pred,
            self.sens_train,
            sens_classes=self.sens_classes,
            dataset=self.sens_multi_dataset,
        )
        f1_score_val = f1_score(self.y_train, y_pred)
        accuracy_score_val = accuracy_score(self.y_train, y_pred)

        print("Train F1: ", round(f1_score_val, 3))
        print("Train EO: ", round(eo_val, 5) * 100)
        print("Train Accuracy: ", round(accuracy_score_val, 3))
        print()

        train_f1 = f1_score_val
        train_eo = eo_val
        train_acc = accuracy_score_val

        return (
            val_f1,
            val_eo,
            val_acc,
            test_f1,
            test_eo,
            test_acc,
            train_f1,
            train_eo,
            train_acc,
        )
