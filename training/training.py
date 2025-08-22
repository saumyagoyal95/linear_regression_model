import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import mlflow
from dotenv import load_dotenv

def load_data(filepath):
    """Load dataset from a CSV file."""
    df = pd.read_csv(filepath)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y

def train_model(X_train, y_train, alpha=0.5, l1_ratio=0.5):
    """Train ElasticNet model."""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name="Training"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    print(f"Dataset: {dataset_name} \nRMSE: {rmse}\nMAE: {mae}\nR2: {r2}")

    # log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    return rmse, mae, r2

def main():
    load_dotenv()  # Load environment variables from .env file
    mlflow.set_tracking_uri(f"http://{os.getenv('MLFLOW_HOST')}:{os.getenv('MLFLOW_PORT')}")

    current_run_name = "training run"
    experiment_name = "Wine Quality Experiment"
    tags = {
        "model": "ElasticNet", 
        "dataset": "Wine Quality",
        "type": "regression",
        "stage": "training"
        }
    
    mlflow.autolog()
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=current_run_name) as run:

        np.random.seed(40)
        data_path = "data/winequality-red.csv"
        X, y = load_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
        alpha = 0.000001
        l1_ratio = 0.05

        model = train_model(X_train, y_train, alpha, l1_ratio)
        print("--- Training Performance ---")
        evaluate_model(model, X_train, y_train, dataset_name="Training")

        # log tags
        mlflow.set_tags(tags)

        #log model on mlflow
        model_info = mlflow.sklearn.log_model(sk_model=model, name="elasticnet", input_example=X_train)

    # # Save the trained model as a pickle file
    # model_path = "artifacts/linear_regression_model.pkl"
    # with open(model_path, "wb") as f:
    #     pickle.dump(model, f)
    # print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    main()