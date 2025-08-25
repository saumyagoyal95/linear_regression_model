from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from app.model.inference import get_prediction
from fastapi import status

app = FastAPI()

class InputValues(BaseModel):
    alcohol: float

class OutputValues(BaseModel):
    predicted_quality: float


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "Successfully Running the server"}

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=OutputValues, status_code=status.HTTP_200_OK)
async def predict_wine_quality(input_values: InputValues) -> Dict[str, OutputValues]:
    predicted_quality = get_prediction(input_values.alcohol)
    return {"predicted_quality": predicted_quality}

