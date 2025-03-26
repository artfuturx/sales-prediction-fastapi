import pandas as pd
from database_connect import get_data_from_db

orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

print(orders_df.head())


#- Aylık veya ürün bazlı satış özet verisinin hazırlanması(def olarak yazalım.)

orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
orders_df["month"] = orders_df["order_date"].dt.to_period("M")


#- Eksik veri kontrolü ve temizliği

print(orders_df.isnull().sum())
print(order_details_df.isnull().sum())
print(products_df.isnull().sum())
print(customers_df.isnull().sum())
print(categories_df.isnull().sum())

orders_df["ship_region"] = orders_df["ship_region"].fillna("UNKNOWN")
print(orders_df)


#- Özellik mühendisliği: Ay bilgisi, ürün fiyatı, müşteri segmentasyonu gibi özellikler üretme

df_final = (
    orders_df[["order_id", "customer_id", "order_date"]]
    .merge(order_details_df, on="order_id", how="left")
    .merge(
        products_df[["product_id", "product_name", "category_id"]], 
        on="product_id", 
        how="left"
    )
)
print(df_final)
print(df_final.info())

