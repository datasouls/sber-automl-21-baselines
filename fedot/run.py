import sys

import pandas as pd

from fedot.api.main import Fedot


def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    timeout = 1

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]

    if task_type in ["binary", "multiclass"]:
        fedot_task_type = 'classification'
    elif task_type == "reg":
        fedot_task_type = 'regression'
    else:
        raise ValueError(f'Task {task_type} not supported')

    automl = Fedot(problem=fedot_task_type, timeout=timeout)
    automl.fit(features=train_data, target=target_col)
    y_pred = automl.predict_proba(features=test_data)

    if task_type == "binary":
        y_pred = pd.DataFrame({target_col: y_pred})
    elif task_type == "multiclass":
        y_pred = pd.DataFrame(y_pred)
    elif task_type == "reg":
        y_pred = pd.DataFrame({target_col: y_pred})

    y_pred.to_csv(output_path, index=None)


if __name__ == "__main__":
    datasets = ['dresses-sales', 'internet-advertisements', 'eucalyptus', 'bioresponse']
    types = ['binary', 'binary', 'multiclass', 'binary']
    for i, dataset in enumerate(datasets):
        sys.argv = ['',
                    f'{types[i]}',
                    f'C:\\Users\\Nikolay\\Downloads\\datasest_to_tests\\{dataset}\\train\\train.csv',
                    f'C:\\Users\\Nikolay\\Downloads\\datasest_to_tests\\{dataset}\\train\\test.csv',
                    f'D:\\res\\out{i}.csv']
        main()
