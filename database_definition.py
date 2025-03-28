# Pandas ile verilerin çekilmesi

# Aylık veya ürün bazlı satış özet verisinin hazırlanması
# Eksik veri kontrolü ve temizliği
# Özellik mühendisliği: Ay bilgisi, ürün fiyatı, müşteri segmentasyonu gibi özellikler üretme



import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import numpy as np
from database_connect import get_data_from_db

def prepare_data_for_ml():
    orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    orders_df["month"] = orders_df["order_date"].dt.month
    orders_df["year"] = orders_df["order_date"].dt.year

    orders_with_details = pd.merge(
        orders_df,
        order_details_df,
        on="order_id",
        how="inner"
    )

    orders_with_products = pd.merge(
        orders_with_details,
        products_df,
        on="product_id",
        how="left"
    )

    full_data = pd.merge(
        orders_with_products,
        categories_df,
        on="category_id",
        how="left"
    )

    full_data.rename(columns={
        'unit_price_x': 'order_price',  # Sipariş anındaki fiyat
        'unit_price_y': 'catalog_price'  # Katalog fiyatı
    }, inplace=True)

    print(full_data)

    #ÖZELLİK FONKSİYONLARIMIZ
    # 1.Her ay için her üründen kaç adet satıldığını gösterecek analiz
    monthly_product_sales = full_data.groupby(['product_id', 'year', 'month']).agg({ # 3 farklı gruplamayı yönetmek için agg kullandım.
        'quantity': 'sum',
        'product_name': 'first',
        'category_name': 'first',
        'order_price': 'mean',
        'catalog_price': 'first',
        'customer_id': 'first'
    }).reset_index()

    monthly_product_sales['month_sin'] = np.sin(2 * np.pi * monthly_product_sales['month']/12) #12. ay ve 1.ay arasında dongüsellik sağlamak için sin ve cos kullanıyorum.
    monthly_product_sales['month_cos'] = np.cos(2 * np.pi * monthly_product_sales['month']/12)
    print(monthly_product_sales)

    #2. Kategori satış miktarlarına göre ranking(en cok satan kategorideyse tekrar satılma ihtimali yüksek.)
    category_sales = monthly_product_sales.groupby('category_name')['quantity'].sum().reset_index()
    category_sales = category_sales.sort_values('quantity', ascending=False)
    category_sales['category_rank'] = range(1, len(category_sales) + 1)

    monthly_product_sales = pd.merge(
        monthly_product_sales,
        category_sales[['category_name', 'category_rank']],
        on='category_name',
        how='left'
    )
    print(monthly_product_sales)

    #3. İndirim oranlarına göre ranking ve ürünlere etkisini analiz
    monthly_product_sales['discount_rate'] = 1 - (monthly_product_sales['order_price'] / monthly_product_sales['catalog_price'])

    product_discounts = monthly_product_sales.groupby('product_id')['discount_rate'].mean().reset_index() #ortalama indirim oranı
    product_discounts = product_discounts.sort_values('discount_rate', ascending=False)
    product_discounts['discount_rank'] = range(1, len(product_discounts) + 1) # inidirim oranlarına göre bir rank oluşturdum

    monthly_product_sales = pd.merge(
        monthly_product_sales,
        product_discounts[['product_id', 'discount_rank']],
        on='product_id',
        how='left'
    )

#3. Benzersiz müşteri sayısı analizine göre tablo oluşturma
    product_customers = full_data.groupby('product_id').size().reset_index(name='unique_customers')

    monthly_product_sales = pd.merge(
        monthly_product_sales,
        product_customers,
        on='product_id',
        how='left'
    )

    print(monthly_product_sales)

    return monthly_product_sales

if __name__ == "__main__":
    df = prepare_data_for_ml()
    df.to_csv("monthly_product_sales.csv", index=False)



    """
    monthly_product_sales (Ana Tablo)
|
├── category_sales (Kategori Sıralaması)
│ └── Eklenen Sütun: category_rank
│
├── product_discounts (İndirim Sıralaması)
│ └── Eklenen Sütun: discount_rank
│
└── product_customers (Müşteri Analizi)
└── Eklenen Sütun: unique_customers

"""