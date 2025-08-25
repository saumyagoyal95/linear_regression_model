# tests/conftest.py
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import app as backend_app # we'll write this factory in a sec

######################### Backend #########################
@pytest.fixture(scope="session")
def fastapi_app() -> FastAPI:
    return backend_app

@pytest.fixture(scope="session")
def backend_client(fastapi_app: FastAPI):
    return TestClient(fastapi_app)
