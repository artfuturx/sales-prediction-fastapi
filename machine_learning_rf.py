import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from database_definition import prepare_segmented_dataframe 
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from eda_utils import remove_outliers_iqr, apply_log_transform


# 1. Veriyi hazırla
df = prepare_segmented_dataframe()

# 2. Aykırı değerleri çıkar 
# df = remove_outliers_iqr(df, 'quantity')
# aykiri degerleri cikardiktan sonra skor dusuyor.

# 3. Log dönüşüm
df = apply_log_transform(df, 'quantity', 'quantity_log')

corr_matrix = df.corr(numeric_only=True)
print(corr_matrix['quantity'].sort_values(ascending=False))


# Özellik ve hedef
feature_cols = [
    'unit_price', 'discount', 'customer_segment', 
    'monthly_segment','product_segment',
    'stock_reorder_interaction'
]

X = df[feature_cols].copy()
y = df['quantity_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Tahmin ve performans
y_pred_rf = model_rf.predict(X_test)
r2_value_rf = r2_score(y_test,y_pred_rf)
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
print(f"RANDOM FOREST - Test RMSE: {rmse_rf:.2f}, Test R2: {r2_value_rf:.2f} ")


#Cross-Validation-Random Forest
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_rf, X, y, cv=5, scoring='r2')
print("CV R² scores:", cv_scores)
print("Mean CV R²:", cv_scores.mean())