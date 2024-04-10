from fastapi import FastAPI
import uvicorn
from data_collection import run
from FinanceData import FinanceData
from keras.models import load_model
from data_precprocessing import preprocess_data
app = FastAPI()


@app.get("/")
def read ():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_value(data:FinanceData):
    input_data=run(data.coin_name,data.coin_id,data.currency)
    input_data,scaler=preprocess_data(input_data)
    model=load_model("/model.h5")
    output=model.predict(input_data)
    output=output.reshape(30,5)
    scaled_output=scaler.inverse_transform(output)
    return {"prediction":scaled_output.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5049)