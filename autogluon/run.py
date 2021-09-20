import os
import sys

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]

    if task_type == "binary":
        metric = "roc_auc"
    elif task_type == "multiclass":
        metric = "roc_auc_ovo_macro"
    elif task_type == "reg":
        metric = 'root_mean_squared_error'

    gluon_predictor = TabularPredictor(
        label=target_col,
        eval_metric=metric,
    )
    gluon_predictor.fit(
        train_data=train,
        time_limit=10
    )

    if task_type == "binary":
        y_pred = gluon_predictor.predict_proba(test, as_pandas=True)[0]
    elif task_type == "multiclass":
        y_pred = gluon_predictor.predict_proba(test, as_pandas=True)
    elif task_type == "reg":
        y_pred = gluon_predictor.predict(test, as_pandas=True)

    y_pred.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
