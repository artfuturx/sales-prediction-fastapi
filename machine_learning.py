import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from database_definition import prepare_segmented_dataframe 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns



df_ml = prepare_segmented_dataframe()

df_ml['log_quantity'] = np.log1p(df_ml['quantity'])  # log(1 + quantity)

print(df_ml.info())

features = [ 'customer_segment','fidelity_segment','product_mean_quantity_log']
target = 'log_quantity'


X = df_ml[features]
y = df_ml[target]

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Tahmin ve performans
y_pred_rf = model_rf.predict(X_test)
r2_value_rf = r2_score(y_test,y_pred_rf)
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
print(f"RANDOM FOREST - Test RMSE: {rmse_rf:.2f}, Test R2: {r2_value_rf:.2f} ")

#desicion tree
model_dt = DecisionTreeRegressor(random_state=42, max_depth=5)  # max_depth ile sınırlayabilirsin
model_dt.fit(X_train, y_train)

#tahmin ve performans
y_pred_dt = model_dt.predict(X_test)
r2_value_dt = r2_score(y_test,y_pred_dt)
rmse_dt = root_mean_squared_error(y_test, y_pred_dt)
print(f"DESICION TREE - Test RMSE: {rmse_dt:.2f}, Test R2: {r2_value_dt:.2f} ")

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

print(f"LINEER REGRESYON - RMSE: {rmse_lr:.2f}, R2: {r2_value_lr:.2f}")


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_lr, X, y, cv=5, scoring='r2')
print("CV R² scores:", cv_scores)
print("Mean CV R²:", cv_scores.mean())

print(df_ml['quantity'].describe())

corr_matrix = df_ml.corr(numeric_only=True)
print(corr_matrix['quantity'].sort_values(ascending=False))


#- Hedef değişken belirleme (örnek:ürün bazlı satış miktarı)
#- Model seçimi (Şu ana kadar öğrendikleriniz)
#- Modelin eğitilmesi ve test edilmesi
#- Model başarım metriklerinin raporlanması (R2, RMSE vs.) –ARGE
#- Eğitilmiş modelin .pkl veya benzeri formatta kaydedilmesi


'''
features = ['unit_price', 'discount', 'customer_segment', 
            'monthly_segment', 'daily_segment','fidelity_segment']
target = 'log_quantity'

'''