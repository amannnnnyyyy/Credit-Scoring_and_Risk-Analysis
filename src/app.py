import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained models using absolute paths
models = {
    'logistic_regression': joblib.load(os.path.join(BASE_DIR, '../notebooks/model/logistic_regression_model.pkl')),
    'random_forest': joblib.load(os.path.join(BASE_DIR, '../notebooks/model/random_forest_model.pkl')),
    'decision_tree': joblib.load(os.path.join(BASE_DIR, '../notebooks/model/decision_tree_model.pkl')),
    'gradient_boosting': joblib.load(os.path.join(BASE_DIR, '../notebooks/model/gradient_boosting_model.pkl'))
}

# Create a FastAPI instance
app = FastAPI()

# Define the input data model based on your features
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
    CountryCode: float 
    Amount: float
    Value: float
    PricingStrategy: float
    FraudResult: float
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: float
    Std_Deviation_Transaction_Amount: float
    Transaction_Hour: float
    Transaction_Day: float
    Transaction_Month: float
    Transaction_Year: float
    model_name: str 

@app.get('/')
def read_root():
    return """
    <html>
        <head>
            <title>Credit Scoring Model API</title>
        </head>
        <body>
            <h1>Welcome to the Credit Scoring Model API</h1>
            <p>Use the button below to navigate to the prediction section.</p>
            <a href="/docs">
                <button style="padding: 10px 20px; font-size: 16px;">Go to Prediction</button>
            </a>
        </body>
    </html>
    """
# Define the prediction endpoint
@app.post('/predict')
def predict(input_data: InputData):
    # Check if the chosen model is available
    if input_data.model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict(exclude={"model_name"})])  # Exclude model_name for prediction
    
    # Make predictions using the chosen model
    model = models[input_data.model_name]
    
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        probab_perc = (f'{(probability[0]*100):.2f}')
        if prediction[0] == 1:
            message = f"The customer is likely to default based on the provided information. Probability of default: {probab_perc}%."
        else:
            message = f"The customer is not likely to default based on the provided information. Probability of default: {probab_perc}%."


        return {
            'model': input_data.model_name,
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'message': message
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


