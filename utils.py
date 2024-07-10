import pandas as pd
import numpy as np
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
)
from typing import Tuple

from aif360.datasets.meps_dataset_panel21_fy2016 import MEPSDataset21


adult_race_to_ohe = {
    "Amer-Indian-Eskimo": [0.0, 0.0, 0.0, 0.0],
    "Asian-Pac-Islander": [1.0, 0.0, 0.0, 0.0],
    "Black": [0.0, 1.0, 0.0, 0.0],
    "Other": [0.0, 0.0, 1.0, 0.0],
    "White": [0.0, 0.0, 0.0, 1.0],
}


# Source: https://www.kaggle.com/code/olcaybolat1/deep-learning-classification-w-pytorch
def load_adult_census(dataset: str = "adult", data_folder: str = "../data"):
    if dataset == "adult":
        df = pd.read_csv("{}/adult.csv".format(data_folder))

    # Replace unknown values with NaN
    replace_chars = ["\n", "\n?\n", "?", "\n?", " ?", "? ", " ? ", " ?\n"]
    if any(char in df.values for char in replace_chars):
        df.replace(replace_chars, np.nan, inplace=True)

    df = df.fillna("Missing")

    df.columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df["occupation"] = df["occupation"].str.strip()

    df["new_occupation"] = df["occupation"].replace(
        {
            "Prof-specialty": "Professional_Managerial",
            "Craft-repair": "Skilled_Technical",
            "Exec-managerial": "Professional_Managerial",
            "Adm-clerical": "Sales_Administrative",
            "Sales": "Sales_Administrative",
            "Other-service": "Service_Care",
            "Machine-op-inspct": "Skilled_Technical",
            "Missing": "Unclassified Occupations",
            "Transport-moving": "Skilled_Technical",
            "Handlers-cleaners": "Service_Care",
            "Farming-fishing": "Service_Care",
            "Tech-support": "Skilled_Technical",
            "Protective-serv": "Professional_Managerial",
            "Priv-house-serv": "Service_Care",
            "Armed-Forces": "Unclassified Occupations",
        }
    )

    df.drop(["occupation"], axis=1, inplace=True)
    df.rename(columns={"new_occupation": "occupation"}, inplace=True)

    data_types = {
        "age": "uint8",
        "workclass": "category",
        "fnlwgt": "int32",
        "education": "category",
        "education-num": "uint8",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": "int32",
        "capital-loss": "int32",
        "hours-per-week": "uint8",
        "native-country": "category",
        "income": "category",
    }
    df = df.astype(data_types)

    # drop education and native country columns
    df.drop(["education"], axis=1, inplace=True)
    df.drop(["native-country"], axis=1, inplace=True)

    race_label_encoder = LabelEncoder()

    df["workclass"] = LabelEncoder().fit_transform(df["workclass"])
    df["marital-status"] = LabelEncoder().fit_transform(df["marital-status"])
    df["occupation"] = LabelEncoder().fit_transform(df["occupation"])
    df["relationship"] = LabelEncoder().fit_transform(df["relationship"])
    race_label_encoder.fit(df["race"])
    df["race"] = race_label_encoder.transform(df["race"])
    df["sex"] = LabelEncoder().fit_transform(df["sex"])

    ohe1 = OneHotEncoder(drop="first")
    ohe2 = OneHotEncoder(drop="first")
    ohe3 = OneHotEncoder(drop="first")
    ohe4 = OneHotEncoder(drop="first")
    ohe5 = OneHotEncoder(drop="first")
    ohe6 = OneHotEncoder(drop="first")

    # Fit and transform the categorical features using one-hot encoding
    workclass_encoded = ohe1.fit_transform(df[["workclass"]]).toarray()
    marital_encoded = ohe2.fit_transform(df[["marital-status"]]).toarray()
    occupation_encoded = ohe3.fit_transform(df[["occupation"]]).toarray()
    relationship_encoded = ohe4.fit_transform(df[["relationship"]]).toarray()
    race_encoded = ohe5.fit_transform(df[["race"]]).toarray()
    sex_encoded = ohe6.fit_transform(df[["sex"]]).toarray()

    # Convert the encoded features to pandas DataFrames
    workclass_array = pd.DataFrame(
        workclass_encoded, columns=ohe1.get_feature_names_out()
    )
    marital_array = pd.DataFrame(marital_encoded, columns=ohe2.get_feature_names_out())
    occupation_array = pd.DataFrame(
        occupation_encoded, columns=ohe3.get_feature_names_out()
    )
    relationship_array = pd.DataFrame(
        relationship_encoded, columns=ohe4.get_feature_names_out()
    )
    race_array = pd.DataFrame(race_encoded, columns=ohe5.get_feature_names_out())
    sex_array = pd.DataFrame(sex_encoded, columns=ohe6.get_feature_names_out())

    # Drop the original categorical features
    df_dropped = df.drop(
        ["workclass", "marital-status", "occupation", "relationship", "race", "sex"],
        axis=1,
    )

    # Concatenate the encoded features with the numerical features
    df_encoded = pd.concat(
        [
            workclass_array,
            marital_array,
            occupation_array,
            relationship_array,
            race_array,
            sex_array,
            df_dropped,
        ],
        axis=1,
    )

    df_encoded["income"] = LabelEncoder().fit_transform(df_encoded["income"])

    X = df_encoded.drop(["income"], axis=1).to_numpy()
    y = df_encoded["income"].to_numpy()

    return X, y


# EOD calculations for experiments where the sensitive attribute is binary (e.g. SEX)
def equal_opp_difference(
    y_true, y_pred, sensi_feat, sens_classes=[0, 1], dataset="none"
):
    if dataset != "none":
        return equal_opp_difference_multi(y_true, y_pred, sensi_feat, dataset)

    error_rates = {}

    for i in sens_classes:
        error_rates[i] = {}
        for j in [0, 1]:
            idx = (sensi_feat == i) & (y_true == j)
            expc = y_pred[idx].mean().item()
            if np.isnan(expc):
                expc = 0.0
            error_rates[i][j] = expc

    # Change this
    tprs = []
    fprs = []
    for cls in sens_classes:
        tprs.append(error_rates[cls][1])
        fprs.append(error_rates[cls][0])

    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)

    return max(tpr_diff, fpr_diff)


# EOD calculations for experiments where the sensitive attribute is non-binary (e.g. more than one RACE (Black, White, Asian, etc.))
def equal_opp_difference_multi(y_true, y_pred, sensi_feat, dataset="adult_race"):
    error_rates = {}

    sens_classes = [False, True]

    if dataset == "adult_race":
        sens_attr = np.array(adult_race_to_ohe["White"])

    for i in sens_classes:
        error_rates[i] = {}
        for j in [0, 1]:
            idx = (
                torch.from_numpy(np.prod(sensi_feat.numpy() == sens_attr, axis=-1) == i)
            ) & (y_true == j)
            expc = y_pred[idx].mean().item()
            if np.isnan(expc):
                expc = 0.0
            error_rates[i][j] = expc

    # Change this
    tprs = []
    fprs = []
    for cls in sens_classes:
        tprs.append(error_rates[cls][1])
        fprs.append(error_rates[cls][0])

    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)

    return max(tpr_diff, fpr_diff)


def preprocess_adult_census(seed: int = 123, scaler: str = "ss"):
    X, y = load_adult_census(dataset="adult")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    columns_to_standardize = [28, 29, 30, 31, 32, 33]

    # Create a StandardScaler object
    scaler = StandardScaler()

    scaler.fit(X_train[:, columns_to_standardize])

    X_train[:, columns_to_standardize] = scaler.transform(
        X_train[:, columns_to_standardize]
    )
    X_val[:, columns_to_standardize] = scaler.transform(
        X_val[:, columns_to_standardize]
    )
    X_test[:, columns_to_standardize] = scaler.transform(
        X_test[:, columns_to_standardize]
    )

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    return X_train, X_test, X_val, y_train, y_test, y_val, 27


def preprocess_bank(seed: int = 42, scaler: str = "mms", data_folder: str = "../data"):
    df = pd.read_csv("{}/bank-full.csv".format(data_folder), sep=";")

    cat_columns_oh = [
        "job",
        "marital",
        "education",
        "contact",
        "poutcome",
    ]

    cat_columns_mms = [
        "age",
        "day",
        "month",
    ]

    num_columns = [
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]

    binary = [
        "default",
        "housing",
        "loan",
    ]

    # Age will be bucketed into 10 buckets
    df["age"] = pd.cut(df["age"], 10, labels=[i for i in range(0, 10)])

    # Convert month from string to int
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month"] = df["month"].map(months)

    # One Hot encode all the cateogrical columns
    oh_encoder = OneHotEncoder(drop="first")
    encoded_data = oh_encoder.fit_transform(df[cat_columns_oh]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out())
    df = df.join(encoded_df)
    df = df.drop(columns=cat_columns_oh)

    # Convert binary columns to 0/1
    df[binary] = df[binary].replace({"no": 0, "yes": 1})

    # convert y to 0/1
    df["y"] = df["y"].replace({"no": 0, "yes": 1})

    # MinMax scale all the mms columns
    mms_scaler = MinMaxScaler(feature_range=(0, 1))
    df[cat_columns_mms] = mms_scaler.fit_transform(df[cat_columns_mms])

    X = df.drop(columns=["y"])
    y = df["y"]

    # Get the index for the sensitive feature (age)
    sens_idx = X.columns.get_loc("age")

    # Create data splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # Use a standard scaler for the remaining columns
    ss = StandardScaler()
    X_train[num_columns] = ss.fit_transform(X_train[num_columns])
    X_test[num_columns] = ss.transform(X_test[num_columns])
    X_val[num_columns] = ss.transform(X_val[num_columns])

    # Convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train.to_numpy()).type(torch.float)
    X_test = torch.from_numpy(X_test.to_numpy()).type(torch.float)
    X_val = torch.from_numpy(X_val.to_numpy()).type(torch.float)
    y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float)
    y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float)
    y_val = torch.from_numpy(y_val.to_numpy()).type(torch.float)

    return X_train, X_test, X_val, y_train, y_test, y_val, sens_idx


def preprocess_compas(
    seed: int = 42, attr="race", scaler: str = "mms", data_folder: str = "../data"
):
    """
    Prepare the data of dataset Compas
    :return: X, Y, input shape and number of classes
    sensitive_param == 3
    """
    X = []
    Y = []
    i = 0
    with open("{}/compas".format(data_folder), "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                # Y.append([1, 0])
                Y.append(0)
            else:
                # Y.append([0, 1])
                Y.append(1)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    if scaler == "mms":
        mms = MinMaxScaler(feature_range=(0, 1))
        # It is expected to know the min-max so this is fine
        X = mms.fit_transform(X)
    else:
        print("DEPRECATED")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    if attr == "race":
        sens_idx = 2
    elif attr == "sex":
        sens_idx = 0
    else:
        raise ValueError("Invalid attr")

    return X_train, X_test, X_val, y_train, y_test, y_val, sens_idx


# Source: https://github.com/Trusted-AI/AIX360/blob/master/examples/tutorials/MEPS.ipynb
def preprocess_meps16(seed: int = 42, scaler: str = "mms"):
    # The MEPS16 tutorial on AIF360 uses the StandardScaler on all features
    # We follow that approach here
    cd = MEPSDataset21()
    df = pd.DataFrame(cd.features)
    X = np.array(df.to_numpy(), dtype=float)
    Y = np.array(cd.labels, dtype=int)
    # Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    X_val = ss.transform(X_val)

    # Convert the NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_val = torch.from_numpy(X_val).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    return X_train, X_test, X_val, y_train, y_test, y_val, 1


def preprocess_default(
    seed: int = 42,
    scaler: str = "mms",
    data_folder: str = "../data",
):
    df = pd.read_csv("{}/UCI_Credit_Card.csv".format(data_folder))
    df = df.rename(columns={"PAY_0": "PAY_1"})
    df = df.drop(columns=["ID"])
    df = df.astype(float)
    all_dependent_columns = df.columns.tolist()
    all_dependent_columns.remove("default.payment.next.month")

    cat_columns_oh = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]  # Categorical columns that need to be one hot
    cat_columns_mms = [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    num_columns = [
        column
        for column in all_dependent_columns
        if column not in cat_columns_oh + cat_columns_mms
    ]  # Numerical columns

    # One Hot encode all the cateogrical columns
    oh_encoder = OneHotEncoder(drop="first")
    encoded_data = oh_encoder.fit_transform(df[cat_columns_oh]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out())
    df = df.join(encoded_df)
    df = df.drop(columns=cat_columns_oh)

    # MinMax scale all the mms columns
    mms_scaler = MinMaxScaler(feature_range=(0, 1))
    df[cat_columns_mms] = mms_scaler.fit_transform(df[cat_columns_mms])

    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    # Get the index for the sensitive feature (sex)
    sens_idx = X.columns.get_loc("SEX_2.0")

    # Create data splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed + 1
    )

    # Use a standard scaler for the remaining columns
    ss = StandardScaler()
    X_train[num_columns] = ss.fit_transform(X_train[num_columns])
    X_test[num_columns] = ss.transform(X_test[num_columns])
    X_val[num_columns] = ss.transform(X_val[num_columns])

    # Convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train.to_numpy()).type(torch.float)
    X_test = torch.from_numpy(X_test.to_numpy()).type(torch.float)
    X_val = torch.from_numpy(X_val.to_numpy()).type(torch.float)
    y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float)
    y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float)
    y_val = torch.from_numpy(y_val.to_numpy()).type(torch.float)

    return X_train, X_test, X_val, y_train, y_test, y_val, sens_idx
