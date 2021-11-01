import sys

import pandas as pd
from supervised.automl import AutoML


def main():
    task_type, train_data, test_data, output_path = sys.argv[1:]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    target_col = list(set(train.columns) - set(test.columns))[0]
    target = train.pop(target_col)

    model = AutoML(total_time_limit=3600, mode="Compete"))
    model.fit(train, target)

    y_pred = model.predict_proba(test)
    if task_type == "binary":
        y_pred = pd.DataFrame({target_col: y_pred[:, 0]})
    elif task_type == "multiclass":
        y_pred = pd.DataFrame(y_pred)
    elif task_type == "reg":
        y_pred = pd.DataFrame({target_col: y_pred})
    y_pred.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()

