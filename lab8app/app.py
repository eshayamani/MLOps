from fastapi import FastAPI
import mlflow
from pydantic import BaseModel

app = FastAPI(
    title="Model Scoring API",
    description="API to get Real-Time Model Predictions",
    version="0.1",
)

class RequestBody(BaseModel):
    input_data: list

@app.on_event('startup')
def load_model():
    global model
    model_path = '/Users/apple/Desktop/MLOps/MLOps/lab8app/random_forest_model'
    model = mlflow.sklearn.load_model(f"file://{model_path}")
    print(f"Model loaded")

@app.post('/predict')
def predict(data: RequestBody):
    input_data = [data.input_data]
    prediction = model.predict(input_data)
    return {'prediction': prediction[0]}
