import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define categories and product names
categories = {
    'Electronics': ['DSLR Camera', 'Smartphone', 'Power Adapter', 'Bluetooth Speaker', 'Wireless Headphones', 'Smartwatch', 'Laptop', 'Tablet', 'Earbuds'],
    'Clothing': ['Cotton Shirt', 'Denim Trousers', 'Floral Skirt', 'Evening Dress', 'Running Shoes', 'High Heels', 'Jacket', 'Sweater'],
    'Beauty': ['Sunscreen SPF50', 'Night Cream', 'Eye Cream', 'Facewash', 'Lip Balm', 'Moisturizer', 'Serum', 'Toner'],
    'Sports': ['Cricket Bat', 'Soccer Ball', 'Batting Gloves', 'Tennis Racket', 'Yoga Mat', 'Dumbbells', 'Badminton Shuttlecock'],
    'Books': ['Science Textbook', 'Earth Science Guide', 'Philosophy Essays', 'Fiction Novel', 'History Book', 'Poetry Collection', 'Math Workbook'],
    'Home': ['LED Lamp', 'Cotton Blanket', 'Coffee Table', 'Kitchen Mixer', 'Vacuum Cleaner', 'Bedside Clock', 'Cushion Set']
}

# Define description templates
description_templates = {
    'Electronics': [
        'High-performance {name} with cutting-edge technology.',
        'Sleek {name} for modern connectivity.',
        'Durable {name} with long-lasting battery.',
        'Compact {name} for tech enthusiasts.'
    ],
    'Clothing': [
        'Stylish {name} for all occasions.',
        'Comfortable {name} with premium fabric.',
        'Trendy {name} for everyday wear.',
        'Elegant {name} for a chic look.'
    ],
    'Beauty': [
        'Effective {name} for radiant skin.',
        'Gentle {name} for daily care.',
        'Nourishing {name} with natural ingredients.',
        'Premium {name} for skincare routines.'
    ],
    'Sports': [
        'High-performance {name} for athletes.',
        'Durable {name} for intense training.',
        'Lightweight {name} for active lifestyles.',
        'Reliable {name} for sports enthusiasts.'
    ],
    'Books': [
        'Engaging {name} for knowledge seekers.',
        'Insightful {name} with rich content.',
        'Captivating {name} for avid readers.',
        'Informative {name} for learning.'
    ],
    'Home': [
        'Modern {name} for home convenience.',
        'Elegant {name} for cozy spaces.',
        'Functional {name} with sleek design.',
        'Durable {name} for daily use.'
    ]
}

# Generate products
product_ids = range(1, 51)  # ~50 products
product_data = []
for pid in product_ids:
    category = random.choice(list(categories.keys()))
    name = random.choice(categories[category])
    price = round(random.uniform(5.0, 200.0), 2)
    description = random.choice(description_templates[category]).format(name=name)
    product_data.append({
        'productId': pid,
        'productName': name,
        'productCategory': category,
        'price': price,
        'productDescription': description
    })
products = pd.DataFrame(product_data)

# Generate users
user_ids = range(1, 31)  # ~30 users
user_names = [f"User_{i}" for i in user_ids]
users = pd.DataFrame({'userId': user_ids, 'userName': user_names})

# Generate interactions (~100 rows)
num_interactions = 100
interaction_ids = range(1, num_interactions + 1)
start_date = datetime(2024, 1, 1)
interactions = []
for iid in interaction_ids:
    product = products.sample(1, random_state=iid).iloc[0]
    user = users.sample(1, random_state=iid).iloc[0]
    qty = random.randint(1, 5)
    bought = random.choice([True, False])
    view = random.choice([True, False]) if not bought else True  # Viewed if bought
    return_status = random.choice([True, False]) if bought else False  # Return only if bought
    review = random.choice([
        "Great product, highly recommend!",
        "Good quality but pricey.",
        "Not as expected, average.",
        "Excellent value for money!",
        "Really disappointed, poor quality.",
        "Fantastic purchase, love it!",
        "Decent product, could be better.",
        "Amazing features, worth every penny!",
        "Broke after a week, not great.",
        "Pretty good, fast delivery!",
        None
    ]) if random.random() > 0.3 else None
    interactions.append({
        'productId': product['productId'],
        'productName': product['productName'],
        'productCategory': product['productCategory'],
        'price': product['price'],
        'qtyBought': qty if bought else 0,
        'productDescription': product['productDescription'],
        'userId': user['userId'],
        'userName': user['userName'],
        'reviewId': f"REV_{iid}",
        'review': review,
        'boughtStatus': bought,
        'viewStatus': view,
        'returnStatus': return_status
    })

# Create interactions DataFrame
interactions = pd.DataFrame(interactions)

# Compute avgRating per product
product_ratings = []
for pid in products['productId']:
    product_reviews = interactions[interactions['productId'] == pid]['review']
    ratings = []
    for review in product_reviews:
        if review:
            # Assign rating based on review sentiment
            if "great" in review.lower() or "excellent" in review.lower() or "fantastic" in review.lower() or "amazing" in review.lower():
                ratings.append(random.uniform(4.0, 5.0))
            elif "good" in review.lower() or "decent" in review.lower() or "pretty" in review.lower():
                ratings.append(random.uniform(3.0, 4.0))
            elif "poor" in review.lower() or "disappointed" in review.lower() or "broke" in review.lower() or "not great" in review.lower():
                ratings.append(random.uniform(1.0, 2.5))
            else:
                ratings.append(random.uniform(2.5, 3.5))  # Neutral
    avg_rating = round(np.mean(ratings) if ratings else random.uniform(3.0, 4.0), 2)
    product_ratings.append({'productId': pid, 'avgRating': avg_rating})
ratings_df = pd.DataFrame(product_ratings)
interactions = interactions.merge(ratings_df, on='productId', how='left')

# Save to CSV
interactions.to_csv('interactions.csv', index=False)

print(f"Generated: {len(products)} unique products, {len(users)} unique users, {len(interactions)} interactions")