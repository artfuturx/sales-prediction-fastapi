#FastAPI  ile temel yapı kurulumu Aşağıdaki uç noktaların oluşturulması:
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


from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import joblib  
from database_definition import prepare_data_for_ml
from machine_learning import train_linear_regression_model


app = FastAPI(title="Sales Prediction API", description="Sales Prediction API with Lineer Regresion")


class PredictionInput(BaseModel):
    product_id: int
    year: int
    month: int
    order_price: float
    catalog_price: float
    category_rank: int


def load_or_train_model():

    return joblib.load('linear_regression_model.pkl')

def get_model():
    return load_or_train_model()







