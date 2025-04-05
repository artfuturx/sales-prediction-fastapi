import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from database_definition import prepare_segmented_dataframe
from eda_utils import apply_log_transform


# Model eğitimi ve kaydı
def train_and_save_model(df, model_path="model.pkl"):
    feature_cols = [
        'monthly_segment', 'product_segment', 'product_mean_spent',
        'stock_reorder_interaction', 'category_rank', 'has_discount'
    ]
    X = df[feature_cols]
    y = df['total_spent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, "scaler.pkl")  # Tahmin fonksiyonunda kullanılacak

    return model, scaler


# Özellik vektörü oluşturma
def build_feature_vector(df, product_id, customer_id, order_date):
    try:
        parsed_date = pd.to_datetime(order_date, dayfirst=True, errors='coerce')
        order_month = parsed_date.month if not pd.isna(parsed_date) else None
    except:
        order_month = None

    try:
        product_segment = df[df['product_id'] == product_id]['product_segment'].mode()[0]
    except IndexError:
        product_segment = 1

    try:
        customer_segment = df[df['customer_id'] == customer_id]['customer_segment'].mode()[0]
    except IndexError:
        customer_segment = 1

    try:
        if order_month is not None:
            segment_mode = df[df['order_month_num'] == order_month]['monthly_segment'].mode()
            monthly_segment = segment_mode.iloc[0] if not segment_mode.empty else 1
        else:
            monthly_segment = 1
    except:
        monthly_segment = 1

    try:
        product_mean_spent = df[df['product_id'] == product_id]['product_mean_spent'].mean()
        if np.isnan(product_mean_spent):
            product_mean_spent = 0
    except:
        product_mean_spent = 0

    try:
        category_rank = df[df['product_id'] == product_id]['category_rank'].mode()[0]
    except:
        category_rank = 1

    try:
        units_in_stock = df[df['product_id'] == product_id]['units_in_stock'].mode()[0]
    except:
        units_in_stock = 0
    try:
        reorder_level = df[df['product_id'] == product_id]['reorder_level'].mode()[0]
    except:
        reorder_level = 0
    
    try:
        discount = df[df['product_id'] == product_id]['discount'].mode()[0]
        has_discount = 1 if discount > 0 else 0
    except:
        discount = 0
        has_discount = 0

    stock_reorder_interaction = units_in_stock * reorder_level

    features = {
        'monthly_segment': monthly_segment,
        'product_segment': product_segment,
        'product_mean_spent': product_mean_spent,
        'stock_reorder_interaction': stock_reorder_interaction,
        'category_rank': category_rank,
        'has_discount': has_discount
    }

    return pd.DataFrame([features])


# Tahmin fonksiyonu
def model_predict(df, product_id, customer_id, order_date):
    input_df = build_feature_vector(df, product_id, customer_id, order_date)
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return int(round(prediction[0]))


# Ana çalışma bloğu
if __name__ == "__main__":
    # Veriyi hazırla
    df = prepare_segmented_dataframe()

    # Modeli eğit ve kaydet
    model, scaler = train_and_save_model(df)

    # Örnek tahmin
    example_product = df['product_id'].iloc[0]
    example_customer = df['customer_id'].iloc[0]
    example_date = '15/03/1997'

    prediction = model_predict(df, example_product, example_customer, example_date)
    print(f"Tahmin edilen miktar: {prediction}")

