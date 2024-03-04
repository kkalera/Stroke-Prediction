import joblib
from fastapi import FastAPI
from typing import List
import pandas as pd

app = FastAPI()


def make_prediction(data: List[float]):
    with open("models/deployment/deployment_model.pkl", "rb") as f:
        model = joblib.load(f)

    column_names = ["age", "avg_glucose_level", "bmi"]
    data_df = pd.DataFrame([data], columns=column_names)
    prediction = model.predict(data_df)
    print(model.named_steps["preprocessor"].get_feature_names_out())
    return bool(prediction[0])


@app.post("/predict")
def predict(data: List[float]):
    return {"prediction": make_prediction(data)}
