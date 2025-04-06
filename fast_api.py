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
    df["total"] = df["quantity"] * df["unit_price"] * (1 - df["discount"])
    summary = df.groupby("product_id")["total"].sum().reset_index()
    summary = summary.rename(columns={"total": "total_spent"})
    return summary.to_dict(orient="records")

# Tahmin için istek modeli
class PredictRequest(BaseModel):
    product_id: int
    customer_id: str
    order_date: str

# /predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    if request.product_id not in df['product_id'].values:
        return {"error": f"Geçersiz ürün ID: {request.product_id}"}
    if request.customer_id not in df['customer_id'].values:
        return {"error": f"Geçersiz müşteri ID: {request.customer_id}"}
    try:
        pd.to_datetime(request.order_date, dayfirst=True)
    except:
        return {"error": "Geçersiz tarih formatı. Lütfen GG/AA/YYYY formatında girin."}

    prediction = model_predict(...)
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
