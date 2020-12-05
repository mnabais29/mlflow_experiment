import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

def eval_metrics(actual,pred):
    f1 = f1_score(actual, pred)
    acc = accuracy_score(actual,pred)
    roc_auc = roc_auc_score(actual, pred)
    return f1,acc,roc_auc

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine-quality csv file from the URL
    #csv_url = (
    #    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    #)
    try:
        data = pd.read_csv('..\mlflow_experiment\winequality-red.csv',sep=";") #pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    # Now seperate the dataset as response variable and feature variabes
    x = data.drop('quality', axis=1)
    y = data['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    with mlflow.start_run(experiment_id=RFC):
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(x_train,y_train)

        prediction = rfc.predict(x_test)

        (f1,acc,roc_auc) = eval_metrics(y_test,prediction)

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            mlflow.sklearn.log_model(rfc, "model", registered_model_name="RFCWineModel")
        else:
            mlflow.sklearn.log_model(rfc, "model")

