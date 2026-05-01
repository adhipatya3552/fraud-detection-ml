from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("models/model.pkl", "rb"))

# ✅ Structured input
class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: Transaction):
    values = np.array(data.features).reshape(1, -1)

    prob = model.predict_proba(values)[0][1]
    prediction = int(prob > 0.5)

    return {
        "fraud": prediction,
        "probability": float(prob),
        "message": "High risk transaction" if prediction == 1 else "Normal transaction"
    }