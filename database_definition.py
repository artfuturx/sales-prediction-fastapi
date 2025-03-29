import pandas as pd
from database_connect import get_data_from_db
import numpy as np

def prepare_segmented_dataframe():
    orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

    #kullanilmasi on gorulen sutunlarin tablolardan cekimi
    df_final = (
        orders_df[["order_id", "customer_id", "order_date"]]
        .merge(order_details_df, on="order_id", how="left")
        .merge(products_df[["product_id", "category_id",'units_in_stock','reorder_level']], on="product_id", how="left")
    )

    # musteri bazli satis musteri segmentasyonu
    customer_sales = df_final.groupby('customer_id')['quantity'].mean().reset_index(name='avg_quantity')
    customer_sales['customer_segment'] = pd.qcut(
        customer_sales['avg_quantity'].rank(method='first'),
        q=44,
        labels=range(1, 45)
    ).astype(int)

    df_final = df_final.merge(customer_sales[['customer_id', 'customer_segment']], on='customer_id', how='left')

    #musteri segmentasyonu 2
    df_final['total_spend'] = df_final['unit_price'] * df_final['quantity']
    customer_value = df_final.groupby('customer_id')['total_spend'].sum().reset_index(name='total_spent')
    
    customer_value['fidelity_segment'] = pd.qcut(
    customer_value['total_spent'].rank(method='first'),
    q=5,
    labels=range(1, 6)
    ).astype(int)
    df_final = df_final.merge(customer_value[['customer_id',
                                              'fidelity_segment']], on='customer_id', how='left')

    # urun bazli satis ve urun segmentasyonu
    product_sales = df_final.groupby('product_id')['quantity'].sum().reset_index(name='avg_quantity')
    product_sales['product_segment'] = pd.qcut(
        product_sales['avg_quantity'].rank(method='first'),
        q=77,
        labels=range(1, 78)
    ).astype(int)

    df_final = df_final.merge(product_sales[['product_id',
                                             'product_segment']], on='product_id', how='left')

    #urun bazli 2
    mean_quantity = df_final.groupby('product_id')['quantity'].mean()
    df_final['product_mean_quantity'] = df_final['product_id'].map(mean_quantity)
    df_final['product_mean_quantity_log'] = np.log1p(df_final['product_mean_quantity'])
    
    # One-hot encoding for product_id
    product_dummies = pd.get_dummies(df_final['product_id'], prefix='product')

    # Ana dataframe'e birleştir
    df_final = pd.concat([df_final, product_dummies], axis=1)


    # aylarin numaralarinin alinmasi
    df_final['order_month_num'] = pd.to_datetime(df_final['order_date']).dt.month

    # aylara gore satis ve segmentasyon
    monthly_sales = df_final.groupby('order_month_num')['quantity'].mean().reset_index(name='avg_quantity')
    monthly_sales['monthly_segment'] = pd.qcut(
        monthly_sales['avg_quantity'].rank(method='first'),
        q=12,
        labels=range(1, 13)
    ).astype(int)

    df_final = df_final.merge(monthly_sales[['order_month_num',
                                            'monthly_segment']], on='order_month_num', how='left')


    df_final['order_day_num'] = pd.to_datetime(df_final['order_date']).dt.day
    daily_sales = df_final.groupby('order_day_num')['quantity'].mean().reset_index(name='avg_quantity')

    daily_sales['daily_segment'] = pd.qcut(
    daily_sales['avg_quantity'].rank(method='first'),
    q=15,
    labels=range(1, 16)
    ).astype(int)

    df_final = df_final.merge(daily_sales[['order_day_num',
                                           'daily_segment']], on='order_day_num', how='left')

    df_final['has_discount'] = (df_final['discount'] > 0).astype(int)

    return df_final
