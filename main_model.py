import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from database_definition import prepare_segmented_dataframe
from eda_utils import apply_log_transform

# 1. Veriyi hazirla
df = prepare_segmented_dataframe()

# 2. Log donusum uygula (aykiri temizleme yapilmiyor)
df = apply_log_transform(df, 'quantity', 'quantity_log')

# 3. Ozellikler ve hedef
feature_cols = [
    'unit_price', 'discount', 'customer_segment', 
    'monthly_segment','product_segment',
    'stock_reorder_interaction'
]

X = df[feature_cols].copy()
y = df['quantity_log']

# 4. Scale et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Modeli kur ve egit
model = LinearRegression()
model.fit(X_scaled, y)

# 8. Model ve scaler kaydet
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 9. Tahmin icin gerekli feature vektoru olusturma fonksiyonu
def build_feature_vector(df, product_id, customer_id, order_date, units_in_stock, reorder_level):
    order_month = pd.to_datetime(order_date).month

    try:
        product_segment = df[df['product_id'] == product_id]['product_segment'].mode()[0]
    except IndexError:
        product_segment = 1

    try:
        customer_segment = df[df['customer_id'] == customer_id]['customer_segment'].mode()[0]
    except IndexError:
        customer_segment = 1

    try:
        monthly_segment = df[df['order_month_num'] == order_month]['monthly_segment'].mode()[0]
    except IndexError:
        monthly_segment = 1

    try:
        unit_price = df[df['product_id'] == product_id]['unit_price'].mode()[0]
    except:
        unit_price = 0

    try:
        discount = df[df['product_id'] == product_id]['discount'].mode()[0]
    except:
        discount = 0

    stock_reorder_interaction = units_in_stock * reorder_level

    features = {
        'unit_price': unit_price,
        'discount': discount,
        'customer_segment': customer_segment,
        'monthly_segment': monthly_segment,
        'product_segment': product_segment,
        'stock_reorder_interaction': stock_reorder_interaction
    }

    return pd.DataFrame([features])

# 10. Tahmin fonksiyonu

def model_predict(product_id, customer_id, order_date, units_in_stock, reorder_level):
    input_df = build_feature_vector(df, product_id, customer_id, order_date, units_in_stock, reorder_level)
    input_scaled = scaler.transform(input_df)
    log_pred = model.predict(input_scaled)
    return int(round(np.expm1(log_pred[0])))
