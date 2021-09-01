import os
import sys

import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]

    laml = TabularAutoML(task=Task(task_type), timeout=10)
    laml.fit_predict(train_data=train, roles={"target": target_col})
    predict = laml.predict(test)

    if task_type in ["binary", "reg"]:
        predict = pd.DataFrame({target_col: predict.data.ravel()})

    elif task_type == "multiclass":
        predict = pd.DataFrame({target_col: np.argmax(predict.data, axis=1)})

    predict.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
