from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import psycopg2
from database_definition import prepare_segmented_dataframe
from database_connect import get_data_from_db
from main_model import build_feature_vector, model_predict, train_and_save_model

# Başlangıçta verileri hazırla
df = prepare_segmented_dataframe()
orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

app = FastAPI(
    title="Sales Predict API",
    description="Northwind DB satış miktarı tahmin servisi"
)

# /products endpoint
@app.get("/products")
def get_products():
    return products_df.to_dict(orient="records")

# /sales_summary endpoint
@app.get("/sales_summary")
def sales_summary():
    summary = df.groupby('product_id')['quantity'].sum().reset_index()
    summary = summary.rename(columns={'quantity': 'total_quantity'})
    return summary.to_dict(orient="records")

# Tahmin için istek modeli
class PredictRequest(BaseModel):
    product_id: int
    customer_id: str
    order_date: str

# /predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model_predict(
        df=df,
        product_id=request.product_id,
        customer_id=request.customer_id,
        order_date=request.order_date
    )
    return {"prediction": prediction}

# /retrain endpoint
@app.post("/retrain")
def retrain():
    df = prepare_segmented_dataframe()
    train_and_save_model(df)
    return {"message": "Model başarıyla tekrar eğitildi."}


#python -m uvicorn fast_api:app --reload
#http://localhost:8000/docs
#http://localhost:8000/redoc
