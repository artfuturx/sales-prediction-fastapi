import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from database_connect import get_data_from_db



orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()


#- Hedef değişken belirleme (örnek:ürün bazlı satış miktarı)
#- Model seçimi (Şu ana kadar öğrendikleriniz)
#- Modelin eğitilmesi ve test edilmesi
#- Model başarım metriklerinin raporlanması (R2, RMSE vs.) –ARGE
#- Eğitilmiş modelin .pkl veya benzeri formatta kaydedilmesi


