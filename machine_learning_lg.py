import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from database_definition import prepare_segmented_dataframe 
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from eda_utils import remove_outliers_iqr, apply_log_transform
from sklearn.model_selection import cross_val_score


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

#lineer regresyon
# Ölçekleme işlemi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# Tahmin ve performans
y_pred_lr = model_lr.predict(X_test_scaled)
r2_value_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(root_mean_squared_error(y_test, y_pred_lr))

# Cross-validation skorları (R²)
cv_scores = cross_val_score(model_lr, X, y, cv=5, scoring='r2')

# 1. Performans metrikleri (test seti ve CV ortalaması)
summary_df = pd.DataFrame({
    "Metric": ["RMSE (Test)", "R² (Test)", "Mean CV R²"],
    "Score": [rmse_lr, r2_value_lr, cv_scores.mean()]
})

# 2. Cross-validation her fold skoru
cv_details_df = pd.DataFrame({
    "Fold": [f"Group {i+1}" for i in range(len(cv_scores))],
    "R² Score": cv_scores
})

# Çıktı
print("📊 Model Performans Özeti:")
print(summary_df)
print("\n🔁 Cross-Validation:")
print(cv_details_df)