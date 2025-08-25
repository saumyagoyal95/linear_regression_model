# tests/conftest.py
import pandas as pd
import pytest

@pytest.fixture
def sample_data():
    data = {
        'fixed acidity': [7.4, 7.8, 7.8],
        'volatile acidity': [0.7, 0.88, 0.76],
        'citric acid': [0.0, 0.0, 0.04],
        'residual sugar': [1.9, 2.6, 2.3],
        'chlorides': [0.076, 0.098, 0.092],
        'free sulfur dioxide': [11.0, 25.0, 15.0],
        'total sulfur dioxide': [34.0, 67.0, 54.0],
        'density': [0.9978, 0.9968, 0.997],
        'pH': [3.51, 3.2, 3.26],
        'sulphates': [0.56, 0.68, 0.65],
        'alcohol': [9.4, 9.8, 10.0],
        'quality': [5, 5, 5]
    }
    df = pd.DataFrame(data)
    return df