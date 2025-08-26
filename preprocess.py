# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from random import randrange
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_adult_ac1():
  #!/usr/bin/env python
  # coding: utf-8

  # data = pd.read_csv("adult.csv")
  train_path = './neufair/data_new/adult/adult.data'
  test_path = './neufair/data_new/adult/adult.test'

  column_names = ['age', 'workclass', 'fnlwgt', 'education',
              'education-num', 'marital-status', 'occupation', 'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
              'native-country', 'income-per-year']
  na_values=['?']

  train = pd.read_csv(train_path, header=None, names=column_names,
                      skipinitialspace=True, na_values=na_values)
  test = pd.read_csv(test_path, header=0, names=column_names,
                     skipinitialspace=True, na_values=na_values)

  df = pd.concat([test, train], ignore_index=True)

  del_cols = ['fnlwgt'] # 'education-num'
  df.drop(labels = del_cols,axis = 1,inplace = True)

  ##### Drop na values
  dropped = df.dropna()
  count = df.shape[0] - dropped.shape[0]
  print("Missing Data: {} rows removed.".format(count))
  df = dropped

  encoders = {}
  cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
  ## Implement label encoder instead of one-hot encoder
  for feature in cat_feat:
      le = LabelEncoder()
      df[feature] = le.fit_transform(df[feature])
      encoders[feature] = le
#    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

  ## Implement label encoder instead of one-hot encoder
  cat_feat = ['race']
  for feature in cat_feat:
      le = LabelEncoder()
      df[feature] = le.fit_transform(df[feature])
      encoders[feature] = le
    
  bin_cols = ['capital-gain', 'capital-loss']
  for feature in bin_cols:
      bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
      df[feature] = bins.fit_transform(df[[feature]])
      encoders[feature] = bins
#    df = df[columns]
  label_name = 'income-per-year'

  favorable_label = 1
  unfavorable_label = 0
  favorable_classes=['>50K', '>50K.']

  pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
  df.loc[pos, label_name] = favorable_label
  df.loc[~pos, label_name] = unfavorable_label

  X = df.drop(labels = [label_name], axis = 1, inplace = False)
  y = df[label_name]

  seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
  return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'), encoders)