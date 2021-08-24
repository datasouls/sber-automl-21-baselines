import os
import sys

import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    ## run model for carbon monoxide
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    laml = TabularAutoML(task=Task(task_type), timeout=10)
    if task_type == "binary":
        laml.fit_predict(train_data=train, roles={"target": "DEATH_EVENT"})

        predict = laml.predict(test)
        predict = pd.DataFrame({"DEATH_EVENT": predict.data.ravel()})

    elif task_type == "multiclass":
        laml.fit_predict(train_data=train, roles={"target": "Species"})

        predict = laml.predict(test)
        predict = pd.DataFrame({"Species": np.argmax(predict.data, axis=1)})

    predict.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
