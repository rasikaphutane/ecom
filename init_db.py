import sqlite3
import pandas as pd

# Load data from interactions.csv
interactions = pd.read_csv('interactions.csv')

# Create products table (unique products)
products = interactions[['productId', 'productName', 'productCategory', 'price', 'productDescription', 'avgRating']].drop_duplicates()

# Create users table with interactions (productIds where boughtStatus=True)
users = interactions[interactions['boughtStatus']].groupby('userId').agg({
    'userName': 'first',
    'productId': lambda x: ','.join(map(str, x))
}).reset_index()
users = users.rename(columns={'productId': 'interactions'})
# Add users with no bought interactions (empty interactions)
all_users = interactions[['userId', 'userName']].drop_duplicates()
users = all_users.merge(users, on=['userId', 'userName'], how='left')
users['interactions'] = users['interactions'].fillna('')

# Create SQLite database
conn = sqlite3.connect('ecommerce.db')
cursor = conn.cursor()

# Create products table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    productId INTEGER PRIMARY KEY,
    productName TEXT,
    productCategory TEXT,
    price REAL,
    productDescription TEXT,
    avgRating REAL
)
''')

# Create users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    userId INTEGER PRIMARY KEY,
    userName TEXT,
    interactions TEXT
)
''')

# Create interactions table
cursor.execute('''
CREATE TABLE IF NOT EXISTS interactions (
    productId INTEGER,
    productName TEXT,
    productCategory TEXT,
    price REAL,
    qtyBought INTEGER,
    productDescription TEXT,
    userId INTEGER,
    userName TEXT,
    reviewId TEXT,
    review TEXT,
    boughtStatus BOOLEAN,
    viewStatus BOOLEAN,
    returnStatus BOOLEAN,
    avgRating REAL
)
''')

# Insert data into tables
products.to_sql('products', conn, if_exists='replace', index=False)
users.to_sql('users', conn, if_exists='replace', index=False)
interactions.to_sql('interactions', conn, if_exists='replace', index=False)

# Commit and close
conn.commit()
conn.close()
print("Database initialized with products, users, and interactions tables.")