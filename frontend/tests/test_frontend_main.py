import pytest
from unittest.mock import patch
import requests

def test_streamlit_predict_success():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"predicted_quality": 5.0}

        response = requests.post(
            "http://linear-regression-backend:8000/predict",
            json={"alcohol": 10.0}
        )

        assert response.status_code == 200
        assert response.json() == {"predicted_quality": 5.0}

def test_streamlit_predict_failure():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 500

        response = requests.post(
            "http://linear-regression-backend:8000/predict",
            json={"alcohol": 10.0}
        )

        assert response.status_code == 500