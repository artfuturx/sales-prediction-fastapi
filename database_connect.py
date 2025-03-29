import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:12345@localhost:5432/gyk1"

engine = create_engine(DATABASE_URL)

def get_data_from_db():
    
    orders_df = pd.read_sql("SELECT * FROM orders", engine)
    
    order_details_df = pd.read_sql("SELECT * FROM order_details", engine)
    
    products_df = pd.read_sql("SELECT * FROM products", engine)

    customers_df = pd.read_sql("SELECT * FROM customers", engine)
    
    categories_df = pd.read_sql("SELECT * FROM categories", engine)
    
    
    return orders_df, order_details_df, products_df, customers_df, categories_df

if __name__ == "__main__":
    get_data_from_db()



