# import libraries
import pandas as pd
import pickle
import mlflow
import yaml
from sklearn.metrics import accuracy_score
import os
from urllib.parse import urlparse

# dagshub environments
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/SaketMunda/machine-learning-pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="SaketMunda"
os.environ["MLFLOW_TRACKING_PASSWORD"]="1e55436ca00f97fedf699f260d2143bd1e416799"

# load params
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/SaketMunda/machine-learning-pipeline.mlflow")

    # load the model from the disk
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # log metrics to MLFLOW
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"], params["model"])

