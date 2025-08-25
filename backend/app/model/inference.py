import os
import pickle
import pandas as pd
import mlflow
from dotenv import load_dotenv

load_dotenv()

class ModelHandler:
    """
    Class to handle model loading and inference.
    """

    def __init__(self):
        self.registry_name = os.getenv("MODEL_REGISTRY_NAME", "sklearn-lr-develop")
        self.model_version = os.getenv("MODEL_VERSION", "1")
        self.model_path = os.getenv("MODEL_PATH", "linear_regression_model.pkl")

    def load_model(self):
        if not self.check_model_exists():
            print("Model file not found locally. Loading from registry...")
            self.model = self.load_model_from_registry()
            self.save_model()
        else:
            print("Loading model from local file...")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

    def check_model_exists(self) -> bool:
        """Check if the model pickle file exists in the current path."""
        return os.path.exists(self.model_path)

    def load_model_from_registry(self):
        """Load the model from the MLflow registry."""
        model_uri = f"models:/{self.registry_name}/{self.model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model loaded from registry of mlflow: {self.registry_name}, version: {self.model_version}")
        return model

    def save_model(self):
        """Save the loaded model to a pickle file."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, X):
        """
        Make predictions using the loaded model.
        Args:
            X (array-like or DataFrame): Input features
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)

def get_prediction(input_values: float):
    """
    Generate predictions for wine quality based on input features.
    Args:
        input_values (float): Alcohol value provided by the user.
    Returns:
        float: Predicted wine quality.
    """
    # Define the feature names (update as per your model's training features)
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    # Static values for all features except alcohol
    static_values = {
        'fixed acidity': 8.0,
        'volatile acidity': 0.5,
        'citric acid': 0.3,
        'residual sugar': 2.0,
        'chlorides': 0.08,
        'free sulfur dioxide': 15.0,
        'total sulfur dioxide': 40.0,
        'density': 0.996,
        'pH': 3.3,
        'sulphates': 0.65
    }
    # alcohol = float(input("Enter alcohol value: "))
    alcohol = input_values

    user_input = [
        static_values['fixed acidity'],
        static_values['volatile acidity'],
        static_values['citric acid'],
        static_values['residual sugar'],
        static_values['chlorides'],
        static_values['free sulfur dioxide'],
        static_values['total sulfur dioxide'],
        static_values['density'],
        static_values['pH'],
        static_values['sulphates'],
        alcohol
    ]
    X = pd.DataFrame([user_input], columns=feature_names)
    handler = ModelHandler()
    handler.load_model()
    preds = handler.predict(X)

    return preds[0]

if __name__ == "__main__":
    print("Predicted wine quality:", get_prediction(10.0))