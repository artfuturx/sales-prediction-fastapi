#**FastAPI**  ile temel yapı kurulumu Aşağıdaki uç noktaların oluşturulması:
#Endpoint	Method	Açıklama

#/products	 GET	Ürün listesini döner

#/predict	POST	Tahmin yapılmasını sağlar

#/retrain	            POST	Modeli yeniden eğitir (opsiyonel)

#/sales_summary	GET 	Satış özet verisini döner

#predict uç noktası:

#Kullanıcıdan ürün, tarih ve müşteri bilgilerini alır

#Modeli yükler ve tahmini yapar

#Tahmini sonuç olarak döner

#Swagger dokümantasyonun kontrolü






"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import joblib  # pickle yerine joblib kullanıyoruz
from paste import prepare_data_for_ml

# FastAPI uygulamasını oluştur
app = FastAPI(title="Satış Tahmin API", description="Ürün satış tahmini için API")

# Tahmin isteği için veri modeli
class PredictionRequest(BaseModel):
    product_id: int
    year: int
    month: int

# Model ve sütunları yükle
def load_model():
    model = joblib.load("best_sales_prediction_model.pkl")  # joblib ile yükleme
    model_columns = joblib.load("model_columns.pkl")  # joblib ile yükleme
    return model, model_columns

# Ana sayfa
@app.get("/", tags=["genel"])
def root():
    return {"message": "Satış Tahmin API'sine Hoş Geldiniz!"}

# Ürün listesi endpoint'i
@app.get("/products", tags=["ürünler"])
def get_products():
    data = prepare_data_for_ml()
    products = data[['product_id', 'product_name']].drop_duplicates().to_dict('records')
    return products

# Tahmin endpoint'i
@app.post("/predict", tags=["tahmin"])
def predict_sales(request: PredictionRequest):
    # Modeli yükle
    model, model_columns = load_model()
    data = prepare_data_for_ml()
    
    # Tahmin için veriyi hazırla
    input_data = pd.DataFrame({
        'product_id': [request.product_id],
        'year': [request.year],
        'month': [request.month],
        'category_rank': [data[data['product_id'] == request.product_id]['category_rank'].iloc[0]],
        'discount_rate': [data[data['product_id'] == request.product_id]['discount_rate'].mean()],
        'unique_customers': [data[data['product_id'] == request.product_id]['unique_customers'].iloc[0]]
    })
    
    # One-hot encoding
    input_data = pd.get_dummies(input_data, columns=['product_id'])
    
    # Eksik sütunları doldur
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Sütunları doğru sırayla seç
    input_data = input_data[model_columns]
    
    # Tahmin yap
    prediction = model.predict(input_data)[0]
    
    return {
        "prediction": round(float(prediction), 2),
        "details": {
            "product_id": request.product_id,
            "product_name": data[data['product_id'] == request.product_id]['product_name'].iloc[0],
            "year": request.year,
            "month": request.month
        }
    }

# Satış özeti endpoint'i
@app.get("/sales_summary", tags=["özet"])
def sales_summary(
    year: Optional[int] = Query(None, description="Filtrelemek için yıl"),
    month: Optional[int] = Query(None, description="Filtrelemek için ay")
):
    data = prepare_data_for_ml()
    
    # Filtreleri uygula
    if year:
        data = data[data['year'] == year]
    if month:
        data = data[data['month'] == month]
    
    # Toplam satış özeti
    summary = {
        "total_sales": int(data['quantity'].sum()),
        "top_products": data.groupby(['product_id', 'product_name'])['quantity'].sum().reset_index().sort_values('quantity', ascending=False).head(5).to_dict('records'),
        "category_sales": data.groupby('category_name')['quantity'].sum().reset_index().sort_values('quantity', ascending=False).to_dict('records')
    }
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"
    """