import pandas as pd
from database_connect import get_data_from_db
import numpy as np

def prepare_segmented_dataframe():
    orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

    #kullanilmasi on gorulen sutunlarin tablolardan cekimi
    df_final = (
        orders_df[["order_id", "customer_id", "order_date"]]
        .merge(order_details_df, on="order_id", how="left")
        .merge(products_df[["product_id", "category_id",'units_in_stock','reorder_level']],
               on="product_id", how="left")
    )

    # musteri bazli satis musteri segmentasyonu
    customer_sales = df_final.groupby('customer_id')['quantity'].mean().reset_index(name='avg_quantity')
    customer_sales['customer_segment'] = pd.qcut(
        customer_sales['avg_quantity'].rank(method='first'),
        q=44,
        labels=range(1, 45)
    ).astype(int)

    df_final = df_final.merge(customer_sales[['customer_id', 'customer_segment']], on='customer_id', how='left')


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
    
    df_final['stock_reorder_interaction'] = df_final['units_in_stock'] * df_final['reorder_level']

    
    category_sales = df_final.groupby('category_id')['quantity'].sum().reset_index()
    category_sales = category_sales.sort_values('quantity', ascending=False)
    category_sales['category_rank'] = range(1, len(category_sales) + 1)

    df_final = pd.merge(
        df_final,
        category_sales[['category_id', 'category_rank']],
        on='category_id',
        how='left'
    )
    return df_final
