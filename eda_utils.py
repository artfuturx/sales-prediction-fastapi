import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from database_definition import prepare_segmented_dataframe
import matplotlib.pyplot as plt
import seaborn as sns


def check_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik değer bulunamadı.")


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def apply_log_transform(df, column, new_column_name):
    df[new_column_name] = np.log1p(df[column])
    return df

if __name__ == "__main__":
    df = prepare_segmented_dataframe()


    #Eksik verilerin kontrolu
    check_missing_values(df)

    df = apply_log_transform(df, 'quantity', 'quantity_log')
    df = apply_log_transform(df, 'total_spent', 'total_spent_log')


    # Orijinal ve Log-total Spent dağılımı
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df['total_spent'], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Orijinal total_spent Dağılımı")
    ax1.set_xlabel("total_spent")

    sns.histplot(df['total_spent_log'], kde=True, ax=ax2, color="salmon")
    ax2.set_title("Log Dönüştürülmüş total_spent Dağılımı")
    ax2.set_xlabel("total_spent_log")
    plt.tight_layout()
    plt.show()

    # Orijinal ve Log-Quantity dağılımı
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df['quantity'], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Orijinal Quantity Dağılımı")
    ax1.set_xlabel("quantity")

    sns.histplot(df['quantity_log'], kde=True, ax=ax2, color="salmon")
    ax2.set_title("Log Dönüştürülmüş Quantity Dağılımı")
    ax2.set_xlabel("quantity_log")
    plt.tight_layout()
    plt.show()

    #Aykiri degerlerin gostrilmesi
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['quantity'], color="orange")
    plt.title("Quantity - Aykırı Değerleri Gösteren Boxplot (IQR Öncesi)")
    plt.xlabel("quantity")
    plt.tight_layout()
    plt.show()

    #Aykiri degerlerin gostrilmesi
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['total_spent_log'], color="orange")
    plt.title("total_spent_log - Aykırı Değerleri Gösteren Boxplot")
    plt.xlabel("total_spent_log")
    plt.tight_layout()
    plt.show()

    
    # Tarihi ay bazına çevir:
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M')

    # Aylık satış özeti:
    monthly_sales = df.groupby('month')['total_spent'].sum().reset_index()

    # Ürün bazlı satış:
    product_sales = df.groupby('product_id')[['quantity','total_spent']].sum().reset_index()

    #musteri bazli satis
    customer_sales = df.groupby('customer_id').size().reset_index(name='order_counts')

    # Müşteri segmentasyonu (örnek kural)
    customer_sales['Segment'] = pd.cut(customer_sales['order_counts'], bins=[0, 50, 100, 150],
                       labels=['Bronz', 'Silver','Gold'])
    
    print(" ")
    print("MÜŞTERİ BAZLI SATIŞ ÖZETİ : ")
    print(customer_sales)
    print(" ")
    print("AYLIK SATIŞ ÖZETİ : ")
    print(monthly_sales)
    print(" ")
    print("ÜRÜN BAZLI SATIŞ ÖZETİ :")
    print(product_sales)

    print(" ")
    print("KOLONLAR HAKKINDA GENEL BİLGİ :")
    print(df.info())

    print(" ")
    print("TOTAL SPENT İLE DİĞER KOLONLARIN İLİŞKİSİ (KORELASYON) :")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix['quantity'].sort_values(ascending=False))

