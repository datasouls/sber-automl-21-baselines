import os
import sys

import pandas as pd
import h2o
from h2o.automl import H2OAutoML

def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    h2o.init()

    train = pd.read_csv(train_data)
    test  = pd.read_csv(test_data)

    h2o_train = h2o.H2OFrame(train)
    h2o_test = h2o.H2OFrame(test)

    features = [x for x in h2o_train.columns if x in h2o_test.columns]
    target = list(set(h2o_train.columns) - set(h2o_test.columns))[0]

    h2oaml = H2OAutoML(
        max_runtime_secs=10,
    )

    if task_type in ["binary", "multiclass"]:
        h2o_train[target] = h2o_train[target].asfactor()

    h2oaml.train(
        x=features,
        y=target,
        training_frame=h2o_train
    )

    preds_h2oaml = h2oaml.leader.predict(h2o_test)
    if task_type == "binary":
        preds_h2oaml = preds_h2oaml["p1"]
    elif task_type == "multiclass":
        preds_h2oaml = preds_h2oaml[[i for i in preds_h2oaml.columns if i != 'predict']]
    preds_h2oaml.as_data_frame().to_csv(output_path, index=None)


if __name__ == "__main__":
    main()
