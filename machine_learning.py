"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Önceden hazırlanmış veri setini kullanalım
# İki yöntem var:
# 1. Kaydedilmiş CSV dosyasını okumak:
# monthly_product_sales = pd.read_csv('ml_ready_data.csv')

# 2. Fonksiyonu doğrudan çağırmak (önerilen):
from paste import prepare_data_for_ml

# Veri setini çağırma
monthly_product_sales = prepare_data_for_ml()

print("Veri seti yüklendi:")
print(monthly_product_sales.head())
print(f"Veri seti boyutu: {monthly_product_sales.shape}")

# Eksik değer kontrolü
missing = monthly_product_sales.isnull().sum()
if missing.sum() > 0:
    print("\nEksik değerler tespit edildi:")
    print(missing[missing > 0])
    # Eksik değerleri doldur
    monthly_product_sales = monthly_product_sales.fillna(0)

# Hedef değişken belirleme
# Ürün bazlı satış miktarını tahmin edeceğiz (quantity)
X = monthly_product_sales[['product_id', 'year', 'month', 'category_rank', 'discount_rate', 'unique_customers']]
y = monthly_product_sales['quantity']  # Hedef: ürün bazlı satış miktarı

# Kategorik değişkenleri one-hot encoding ile dönüştürme
X = pd.get_dummies(X, columns=['product_id'], drop_first=True)

# Eğitim ve test verisinin hazırlanması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nEğitim veri seti boyutu: {X_train.shape}")
print(f"Test veri seti boyutu: {X_test.shape}")

# Korelasyon analizi
# Sayısal değişkenlerin hedef değişken ile korelasyonu
numeric_cols = ['year', 'month', 'category_rank', 'discount_rate', 'unique_customers']
corr_with_target = monthly_product_sales[numeric_cols + ['quantity']].corr()['quantity'].sort_values(ascending=False)
print("\nHedef değişken ile korelasyonlar:")
print(corr_with_target)"""