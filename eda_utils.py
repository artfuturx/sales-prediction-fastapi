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
    df = apply_log_transform(df, 'quantity', 'quantity_log')

    check_missing_values(df)

    # Orijinal quantity dağılımı
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