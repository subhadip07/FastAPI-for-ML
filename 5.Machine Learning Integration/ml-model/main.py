from fastapi import FastAPI
from schemas import InputSchema, OutputSchema
from typing import List
from predict import make_batch_predictions, make_prediction

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Welcome to the ML Model Prediction API'}


@app.post('/prediction', response_model=OutputSchema)
def predict(user_input: InputSchema):
    prediction = make_prediction(user_input.model_dump())
    return OutputSchema(predicted_price = round(prediction, 2))


@app.post('/batch_prediction', response_model=List[OutputSchema])
def batch_predict(user_input: List[InputSchema]):
    predictions = make_batch_predictions([x.model_dump() for x in user_input])
    return [OutputSchema(predicted_price = round(predictions, 2)) for prediction in predictions]
