import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {

    'gradient_boosting': joblib.load(os.path.join(BASE_DIR, 'models/gradient_boosting_model.pkl')),
    'logistic_regression': joblib.load(os.path.join(BASE_DIR, 'models/logistic_regression_model.pkl')),
    'random_forest': joblib.load(os.path.join(BASE_DIR, 'models/random_forest_model.pkl')),
    'decision_tree': joblib.load(os.path.join(BASE_DIR, 'models/decision_tree_model.pkl'))
}

# Create a FastAPI instance
app = FastAPI()

origins = [
    "http://localhost:3000",  # React app
    "http://0.0.0.0:8000",    # Backend itself
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)
class InputData(BaseModel):
    TotalRFMS: float
    ProviderId_ProviderId_2: float
    ProviderId_ProviderId_3: float
    ProviderId_ProviderId_4: float
    ProviderId_ProviderId_5: float
    ProviderId_ProviderId_6: float
    ProductId_ProductId_10: float
    ProductId_ProductId_11: float
    ProductId_ProductId_12: float
    ProductId_ProductId_13: float
    ProductId_ProductId_14: float
    ProductId_ProductId_15: float
    ProductId_ProductId_16: float
    ProductId_ProductId_19: float
    ProductId_ProductId_2: float
    ProductId_ProductId_20: float
    ProductId_ProductId_21: float
    ProductId_ProductId_22: float
    ProductId_ProductId_23: float
    ProductId_ProductId_24: float
    ProductId_ProductId_27: float
    ProductId_ProductId_3: float
    ProductId_ProductId_4: float
    ProductId_ProductId_5: float
    ProductId_ProductId_6: float
    ProductId_ProductId_7: float
    ProductId_ProductId_8: float
    ProductId_ProductId_9: float
    ProductCategory_data_bundles: float
    ProductCategory_financial_services: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ChannelId_ChannelId_2: float
    ChannelId_ChannelId_3: float
    ChannelId_ChannelId_5: float
    Amount: float
    Value: float
    PricingStrategy: float
    FraudResult: float
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: float
    # Std_Deviation_Transaction_Amount: float
    Transaction_Hour: float
    Transaction_Day: float
    Transaction_Month: float
    Transaction_Year: float
    model_name: str 
        
    class Config:
        protected_namespaces = ()

app.mount("/static", StaticFiles(directory="."), name="static")



@app.post('/predict')
def predict(input_data: InputData):
    if input_data.model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    input_df = pd.DataFrame([input_data.dict(exclude={"model_name"})])  # Exclude model_name for prediction
    
    model = models[input_data.model_name]
    
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        probab_perc = (f'{(probability[0]*100):.2f}')
        if prediction[0] == 1:
            message = probab_perc
        else:
            message = probab_perc

        return {
            'model': input_data.model_name,
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'message': message
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
