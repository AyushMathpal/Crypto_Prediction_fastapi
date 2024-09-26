from fastapi import FastAPI
import uvicorn
from src.data_collection import run
from src.FinanceData import FinanceData
from keras.models import load_model
from src.data_preprocessing import preprocess_data
app = FastAPI()


@app.get("/")
def read ():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_value(data:FinanceData):
    try:
        input_data=run(data.coin_name,data.coin_id,data.currency)
        print(input_data)
        print(input_data.shape,"Shape")
        input_data,scaler=preprocess_data(input_data)
        model=load_model("src/model.keras")
        output=model.predict(input_data)
        output=output.reshape(30,5)
        scaled_output=scaler.inverse_transform(output)
        return {"prediction":scaled_output.tolist()}
    except:
        print("An exception occured")
        return {"prediction":""}

@app.post("/test")
def test():
    return{"hello world"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5049)