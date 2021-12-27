import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from flaml import AutoML


def processData(data_old):

    data = data_old.copy()
    le = LabelEncoder()
    for column in data.columns:
        print(column)
        try:
            data[column].astype(float)
            data[column] = data[column].fillna(0)
        except:
            try:
                data[column] = pd.to_datetime(data[column])
                data[column + "_day"] = data[column].dt.day
                data[column + "_month"] = data[column].dt.month
                data[column + "_year"] = data[column].dt.year
                data = data.drop(columns=[column])
            except:
                data[column] = data[column].astype(str)
                data[column] = data[column].fillna("No value")
                data[column] = le.fit_transform(data[column])

    return data


def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]
    target = train.pop(target_col)

    data_temp = pd.concat([train, test], ignore_index=True)
    data_temp = processData(data_old=data_temp)
    train, test = data_temp.iloc[: train.shape[0]], data_temp.iloc[train.shape[0] :]

    model = AutoML()
    if task_type in ["binary", "multiclass"]:
        print("Model Fitting")
        model.fit(train, target, task="classification")
    elif task_type == "reg":
        print("Model Fitting")
        model.fit(train, target, task="regression")
    print("Predicting")

    y_pred = pd.DataFrame(model.predict(test), columns=[target_col])
    y_pred.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
