import pandas as pd
import numpy as np
from database_connect import get_data_from_db

def prepare_data_for_ml():
    # Veri setlerini import et
    orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()
    
    # Tarih dönüşümleri
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    orders_df["month"] = orders_df["order_date"].dt.month
    orders_df["year"] = orders_df["order_date"].dt.year
    
    # Temel veri birleştirme
    # Siparişleri ve sipariş detaylarını birleştir
    orders_with_details = pd.merge(
        orders_df,
        order_details_df,
        on="order_id",
        how="inner"
    )
    
    # Ürün ve kategori bilgilerini ekle
    orders_with_products = pd.merge(
        orders_with_details,
        products_df,
        on="product_id",
        how="left"
    )
    
    # Tam veri seti
    full_data = pd.merge(
        orders_with_products,
        categories_df,
        on="category_id",
        how="left"
    )
    
    # Fiyat bilgilerini netleştir
    full_data.rename(columns={
        'unit_price_x': 'order_price',  # Sipariş anındaki fiyat
        'unit_price_y': 'catalog_price'  # Katalog fiyatı
    }, inplace=True)
    
    # Ay-yıl bazında ürün satışlarını hesapla (ana tablo)
    monthly_product_sales = full_data.groupby(['product_id', 'year', 'month']).agg({
        'quantity': 'sum',
        'product_name': 'first',
        'category_name': 'first',
        'order_price': 'mean',
        'catalog_price': 'first'
    }).reset_index()
    
    # 1. KATEGORİ SATIŞLARI SIRALAMA
    category_sales = full_data.groupby('category_name').agg({
        'quantity': 'sum'
    }).reset_index()
    
    # Satış miktarına göre sırala (en çoktan en aza)
    category_sales = category_sales.sort_values('quantity', ascending=False)
    category_sales['category_rank'] = range(1, len(category_sales) + 1)
    
    # Kategori satış rakamını ve sıralamasını ana tabloya ekle
    monthly_product_sales = pd.merge(
        monthly_product_sales,
        category_sales[['category_name', 'category_rank']],
        on='category_name',
        how='left'
    )
    
    # 2. İNDİRİM ORANI ANALİZİ
    monthly_product_sales['discount_rate'] = 1 - (monthly_product_sales['order_price'] / monthly_product_sales['catalog_price'])
    
    # Ürün bazında ortalama indirim oranını hesapla
    product_discounts = monthly_product_sales.groupby('product_id').agg({
        'discount_rate': 'mean'
    }).reset_index()
    
    # İndirim oranına göre sırala
    product_discounts = product_discounts.sort_values('discount_rate', ascending=False)
    product_discounts['discount_rank'] = range(1, len(product_discounts) + 1)
    
    # İndirim sıralamasını ana tabloya ekle
    monthly_product_sales = pd.merge(
        monthly_product_sales,
        product_discounts[['product_id', 'discount_rank']],
        on='product_id',
        how='left'
    )
    
    # 3. MÜŞTERİ ANALİZİ - GÜNCELLENMİŞ
    
    # 3.1 Müşteri sipariş sayısı analizi
    customer_order_counts = orders_df.groupby('customer_id')['order_id'].nunique().reset_index()
    customer_order_counts.columns = ['customer_id', 'order_count']
    
    # Sipariş sayısına göre müşterileri sınıflandır
    # Çeyreklik değerlere göre segmentasyon
    q1_order = customer_order_counts['order_count'].quantile(0.25)
    q3_order = customer_order_counts['order_count'].quantile(0.75)
    
    def categorize_order_count(count):
        if count <= q1_order:
            return 'Low'
        elif count <= q3_order:
            return 'Medium'
        else:
            return 'High'
    
    customer_order_counts['order_frequency_segment'] = customer_order_counts['order_count'].apply(categorize_order_count)
    
    # 3.2 Müşteri toplam harcama analizi
    # Önce her siparişin toplam değerini hesapla
    full_data['line_total'] = full_data['quantity'] * full_data['order_price']
    
    # Müşteri başına toplam harcama
    customer_spending = full_data.groupby('customer_id')['line_total'].sum().reset_index()
    customer_spending.columns = ['customer_id', 'total_spending']
    
    # Harcamaya göre müşterileri sınıflandır
    q1_spending = customer_spending['total_spending'].quantile(0.25)
    q3_spending = customer_spending['total_spending'].quantile(0.75)
    
    def categorize_spending(spending):
        if spending <= q1_spending:
            return 'Low'
        elif spending <= q3_spending:
            return 'Medium'
        else:
            return 'High'
    
    customer_spending['spending_segment'] = customer_spending['total_spending'].apply(categorize_spending)
    
    # 3.3 İki segmenti birleştirerek genel müşteri segmenti oluştur
    # Müşteri segmentlerini birleştir
    customer_segments = pd.merge(
        customer_order_counts,
        customer_spending,
        on='customer_id',
        how='inner'
    )
    
    # Birleşik segment oluştur (frequency ve spending birleşimi)
    segment_mapping = {
        ('Low', 'Low'): 'Low Value',
        ('Low', 'Medium'): 'Occasional Spender',
        ('Low', 'High'): 'Big Spender',
        ('Medium', 'Low'): 'Regular Economy',
        ('Medium', 'Medium'): 'Core Customer',
        ('Medium', 'High'): 'Core Premium',
        ('High', 'Low'): 'Frequent Economy',
        ('High', 'Medium'): 'Enthusiast',
        ('High', 'High'): 'VIP'
    }
    
    customer_segments['customer_segment'] = customer_segments.apply(
        lambda x: segment_mapping.get((x['order_frequency_segment'], x['spending_segment']), 'Unknown'),
        axis=1
    )
    
    # 3.4 Ürün bazında müşteri segmentlerini analiz et
    # Her ürün için hangi segment müşterilerinin satın aldığını belirle
    product_customer_segments = full_data.merge(
        customer_segments[['customer_id', 'customer_segment']],
        on='customer_id',
        how='left'
    )
    
    # Her ürün için en yaygın müşteri segmentini bul
    product_segments = product_customer_segments.groupby(['product_id', 'customer_segment']).size().reset_index(name='count')
    
    # Her ürün için en çok alışveriş yapan segment
    dominant_segments = product_segments.sort_values(['product_id', 'count'], ascending=[True, False])
    dominant_segments = dominant_segments.groupby('product_id').first().reset_index()
    dominant_segments = dominant_segments[['product_id', 'customer_segment']]
    dominant_segments.columns = ['product_id', 'dominant_customer_segment']
    
    # Ana tabloya ekle
    monthly_product_sales = pd.merge(
        monthly_product_sales,
        dominant_segments,
        on='product_id',
        how='left'
    )
    
    # Eski müşteri analizi kodunu da tutalım - benzersiz müşteri sayısı
    product_customers = full_data.groupby('product_id')['customer_id'].nunique().reset_index()
    product_customers.columns = ['product_id', 'unique_customers']
    
    # Müşteri sayısını ana tabloya ekle
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