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


from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import joblib
from database_connect import get_data_from_db
import numpy as np


app = FastAPI(title="Price Predict Api", 
              description="Northwind DB price predict Api's")










"""

joblib.dump(model, "credit_model.pkl")

# 🚀 FastAPI uygulaması
app = FastAPI(title="Credit Approveal API", description="credit approval")

# 📦 Giriş verisi için Pydantic modeli
class Applicant(BaseModel):
    age: int
    income: float  # int de olabilir
    credit_score: int
    has_default: int

# 🔮 Tahmin endpoint'i (tek dekoratör!)
@app.post("/predict", tags=["prediction"])
def predict_approval(applicant: Applicant):
    data_model = joblib.load("credit_model.pkl")
    input_data = [[
        applicant.age,
        applicant.income,
        applicant.credit_score,
        applicant.has_default
    ]]
    prediction = data_model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": result,
        "details": {
            "age": applicant.age,
            "income": applicant.income,
            "credit_score": applicant.credit_score,
            "has_default": applicant.has_default
        }
    }

# 🧠 AR-GE Ödevleri
# Ödev 1 - ARGE : DecisionTrees'de gini yerine alternatif ne kullanılabilir? Farkı nedir?
# Ödev 2 - ARGE : Pydantic ile başka neler yapılabilir? 
# Ödev 3 - ARGE : Faker kütüphanesi ne işe yarar? Detaylı araştırınız."
"""