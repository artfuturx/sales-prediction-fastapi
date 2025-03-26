import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql+psycopg2://sevgi:140216@localhost:5432/northwind"

engine = create_engine(DATABASE_URL)

def get_data_from_db():
    orders_df = pd.read_sql("SELECT * FROM orders", engine)
    
    order_details_df = pd.read_sql("SELECT * FROM order_details", engine)
    
    products_df = pd.read_sql("SELECT * FROM products", engine)

    customers_df = pd.read_sql("SELECT * FROM customers", engine)
    
    categories_df = pd.read_sql("SELECT * FROM categories", engine)
    
    print(f"order_df: {orders_df}")
    print(order_details_df)
    print(products_df)
    print(customers_df)
    print(categories_df)
    
    return orders_df, order_details_df, products_df, customers_df, categories_df

if __name__ == "__main__":
    get_data_from_db()



