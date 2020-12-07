import numpy as np
import pandas as pd
import argparse
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score


import mlflow
import mlflow.sklearn

def eval_metrics(actual,pred):
    f1 = f1_score(actual, pred, average='macro')
    acc = accuracy_score(actual,pred)
    #roc_auc = roc_auc_score(actual, pred, multi_class="ovo", average="macro")
    return f1,acc

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators','-n_est', type=int)
    parser.add_argument('--max_depth','-md', type=int)
    parser.add_argument('--max_features','-mf', type=str)
    parser.add_argument('--random_state','-rs', type=int)
    parser.add_argument('--n_jobs','-nj', type=int)

    args = parser.parse_args()

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    max_features = args.max_features
    random_state = args.random_state
    n_jobs = args.n_jobs

    params = {"n_estimators": n_estimators, "max_depth": max_depth, "max_features": max_features, "random_state": random_state, "n_jobs": n_jobs}

    # Read the wine-quality csv file from the URL
    #csv_url = (
    #    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    #)
    try:
        data = pd.read_csv('..\mlflow_experiment\winequality-red.csv') #pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    # Now seperate the dataset as response variable and feature variabes
    x = data.drop('quality', axis=1)
    y = data['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state= random_state)

    print(y)
    print(y_test)
    print(y_train)

    with mlflow.start_run():

        mlflow.sklearn.autolog()
        rfc = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features, random_state = random_state, n_jobs = n_jobs)
        rfc.fit(x_train,y_train)

        y_pred = rfc.predict(x_test)

        print(classification_report(y_test, y_pred))

        (f1,acc) = eval_metrics(y_test,y_pred)

        mlflow.log_metric("f1", f1)
        mlflow.log_metric("acc", acc)

        #mlflow.log_metric("roc_auc", roc_auc)

        # Model registry does not work with file store
        #if tracking_url_type_store != "file":
            # Register the model
        #    mlflow.sklearn.log_model(rfc, "model", registered_model_name="RFCWineModel")
        #else:
        #    mlflow.sklearn.log_model(rfc, "model")

