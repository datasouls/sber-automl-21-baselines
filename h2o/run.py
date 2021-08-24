import os
import sys

import h2o
from h2o.automl import H2OAutoML

def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    h2o.init()

    h2o_train = h2o.import_file(train_data)
    h2o_test = h2o.import_file(test_data)

    features = [x for x in h2o_train.columns if x not in ["DEATH_EVENT"]]

    h2oaml = H2OAutoML(
        max_runtime_secs=10,
    )

    h2oaml.train(
        x=features, 
        y='DEATH_EVENT', 
        training_frame=h2o_train
    )

    preds_h2oaml = h2oaml.leader.predict(h2o_test)
    preds_h2oaml.as_data_frame().to_csv(output_path, index=None)

if __name__ == "__main__":
    main()
