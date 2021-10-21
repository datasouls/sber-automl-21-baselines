import sys

import pandas as pd

import autosklearn.classification
import autosklearn.regression

from sklearn.preprocessing import LabelEncoder

from itertools import combinations


def processData(data_old):

    data = data_old.copy()
    #Drop duplicates columns
    flag = True
    while flag:
        for pair in combinations(data.columns, 2):
            if data.groupby(pair[0]).agg({pair[1]:'nunique'})[pair[1]].unique().shape[0] == 1 and data.groupby(pair[0]).agg({pair[1]:'nunique'})[pair[1]].unique()[0] == 1:
                try:
                    data[pair[0]].astype(float)
                    data[pair[1]].astype(float)
                    continue
                except:
                    data.drop(columns = pair[0], inplace = True)
                    break
        if pair[1] == data.columns[-1]:
            flag = False
    #Fill null values and label encoding str type columns
    le = LabelEncoder()
    for column in data.columns:
        try:
            data[column].astype(float)
            data[column] = data[column].fillna(0)
        except:
            data[column] = data[column].fillna('No value')
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
    train, test = data_temp.iloc[:train.shape[0]], data_temp.iloc[train.shape[0]:]

    if task_type == 'binary':
        metric = autosklearn.metrics.roc_auc
    elif task_type == 'multiclass':
        metric = autosklearn.metrics.accuracy
    elif task_type == 'reg':
        metric = autosklearn.metrics.root_mean_squared_error

    training_params = {
        'time_left_for_this_task': 30,
        'ensemble_size': 20,
        'ensemble_nbest': 20,
        'metric': metric,
        'seed': 42,
        'memory_limit': None
    }

    if task_type in ['binary', 'multiclass']:
        model = autosklearn.classification.AutoSklearnClassifier(**training_params)
    elif task_type == 'reg':
        model = autosklearn.regression.AutoSklearnRegressor(**training_params)

    print("Training started")
    if task_type in ["binary", "reg"]:
        model.fit(train, target)
        print("Fitting ended")
        y_pred = model.predict_proba(test)[:, 0]
        y_pred = pd.DataFrame({target_col: y_pred})
    elif task_type == "multiclass":
        model.fit(train, target)
        y_pred = model.predict_proba(test)
        y_pred = pd.DataFrame(y_pred)

    print("End of training")
    y_pred.to_csv(output_path, index=None)

if __name__ == "__main__":
    main()