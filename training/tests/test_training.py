import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from training import load_data, train_model, evaluate_model

def test_load_data(sample_data, monkeypatch):
    mock_read_csv = MagicMock(return_value=sample_data)
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    X, y = load_data("mock_path.csv")
    assert not X.empty
    assert not y.empty
    assert "quality" not in X.columns
    assert y.name == "quality"

def test_train_model(sample_data):
    X = sample_data.drop("quality", axis=1)
    y = sample_data["quality"]
    model = train_model(X, y, alpha=0.1, l1_ratio=0.5)
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluate_model(sample_data):
    X = sample_data.drop("quality", axis=1)
    y = sample_data["quality"]
    model = train_model(X, y, alpha=0.1, l1_ratio=0.5)

    with patch("mlflow.log_metric") as mock_log_metric:
        rmse, mae, r2 = evaluate_model(model, X, y, dataset_name="Test")
        assert rmse >= 0
        assert mae >= 0
        assert -1 <= r2 <= 1
        mock_log_metric.assert_called()
