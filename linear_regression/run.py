import os
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]
    target = train.pop(target_col)

    for col in test.columns:
        if test[col].dtype == 'O':
            le = preprocessing.LabelEncoder()
            le.fit(train[col].to_list() + test[col].to_list())

            train[col] = le.transform(train[col])
            test[col]  = le.transform(test[col])
    
    if task_type in ["binary", "reg"]:
        model = LinearRegression()
        model = model.fit(train, target)
        predict = model.predict(test)
        predict = pd.DataFrame({target_col: predict})

    elif task_type == "multiclass":
        lb = preprocessing.LabelBinarizer()
        target_bin = lb.fit_transform(target)
        predict = pd.DataFrame()
        for i in range(target_bin.shape[1]):
            model = LinearRegression()
            model = model.fit(train, target_bin[:, i])

            predict[str(i)] = model.predict(test)

    predict.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
