# NeuFair: Neural Network Fairness Repair with Dropout

This repository contains the source code for "NeuFair: Neural Network Fairness Repair with Dropout" accepted at ACM ISSTA 2024.

## Setup

1. Create the conda environment from the `environment.yml` file.
2. Follow the instructions [here](https://github.com/Trusted-AI/AIF360/blob/main/aif360/data/raw/meps/README.md) to download the MEPS 16 dataset.

## General Instructions

The `saved_models` folder contains the pretrained DNNs used in our experiments for all datasets and seeds. The `data` folder contains the dataset used in our experiments. 

The simulated annealing and random walk repair strategies are implemented in `sa.py`. The `utils.py` script contains utility functions to preprocess the raw datasets and compute the fairness score of the DNN predictions. The `model.py` script defines the DNN architectures for all datasets.

The `sa_experiments` and `random_experiments` contain scripts to run SA and RW strategies for the following datasets and sensitive attributes:
<ol>
    <li> `adult_race`: Adult Census Income dataset with Race </li>
    <li> `adult`: Adult Census Income dataset with Sex </li>
    <li> `bank`: Bank Marketing dataset with Age </li>
    <li> `compas`: Compas Software with Race </li>
    <li> `compas_sex`: Compas Software with Sex </li>
    <li> `default`: Default Credit with Sex </li>
    <li> `meps`: Medical Expenditure with Race </li>
</ol>

To retrain your own models, please use the training scripts provided in `train_models`. Each training script trains models for a given dataset for 10 seeds. The training scripts contain the dataset-specific hyperparameters found after hyperparameter tuning.

## Citation

```
@misc{dasu2024neufairneuralnetworkfairness,
      title={NeuFair: Neural Network Fairness Repair with Dropout}, 
      author={Vishnu Asutosh Dasu and Ashish Kumar and Saeid Tizpaz-Niari and Gang Tan},
      year={2024},
      eprint={2407.04268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.04268}, 
}
```