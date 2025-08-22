import pickle
import numpy as np
import pandas as pd

class ModelInference:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, X):
        """Make predictions using the loaded model.
        Args:
            X (array-like or DataFrame): Input features
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)

def get_prediction(input_values: float):

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
    model_path = "app/model/linear_regression_model.pkl"
    infer = ModelInference(model_path)
    preds = infer.predict(X)
    # print("Predicted wine quality:", preds[0])

    return preds[0]

if __name__ == "__main__":
    print("Predicted wine quality:", get_prediction(10.0))
