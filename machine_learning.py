# Hedef değişken belirleme (örnek: ürün bazlı satış miktarı)
# Eğitim ve test verisinin hazırlanması (train_test_split)
# Model seçimi (Şu ana kadar öğrendikleriniz)
# Modelin eğitilmesi ve test edilmesi
# Model başarım metriklerinin raporlanması (R2, RMSE vs.) –ARGE
# Eğitilmiş modelin .pkl veya benzeri formatta kaydedilmesi



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from database_definition import prepare_data_for_ml
import joblib


def train_linear_regression_model():
    
    df = prepare_data_for_ml()
    print(f"Toplam kayıt sayısı: {len(df)}")
    
    product_avg_sales = df.groupby('product_id')['quantity'].mean().reset_index()
    product_avg_sales.columns = ['product_id', 'product_avg_quantity']
    df = df.merge(product_avg_sales, on='product_id', how='left')
    
    X = df[['year', 'month', 'month_sin', 'month_cos', 'order_price', 
            'catalog_price', 'category_rank', 'discount_rate', 
            'unique_customers', 'product_avg_quantity']]
    
    X = X.fillna(0)

    y = df['quantity']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Eğitim seti boyutu: {X_train.shape[0]}")
    print(f"Test seti boyutu: {X_test.shape[0]}")
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Doğruluğu:")
    print(f"R-kare : {r2}")
    
    coefficients = pd.DataFrame({ #bağımsız deişkenlerin etki katsayıları görme
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coefficients['coef'] = coefficients['Coefficient']
    coefficients = coefficients.sort_values('coef', ascending=False)
    del coefficients['coef']
    
    
    print("\nLinear Regression Katsayıları:")
    print(coefficients)
    
    joblib.dump(model, 'linear_regression_model.pkl') # modeli kaydettim.
    print("Model 'linear_regression_model.pkl' dosyasına kaydedildi.")
    
    return model, coefficients

if __name__ == "__main__":
    model, coefficients = train_linear_regression_model()