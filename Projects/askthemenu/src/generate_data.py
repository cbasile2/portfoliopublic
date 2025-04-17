import pandas as pd
import random
import os

os.makedirs("data", exist_ok=True)

cuisines = ['Italian', 'Mexican', 'Japanese', 'Indian', 'Thai', 'American', 'Chinese']
descriptions = {
    'Italian': 'Fresh pasta, wood-fired pizza, and rich tomato sauces.',
    'Mexican': 'Tacos, enchiladas, spicy salsas, and grilled meats.',
    'Japanese': 'Sushi, ramen, miso soup, and delicate presentation.',
    'Indian': 'Curry dishes, naan bread, vibrant spices, and lentils.',
    'Thai': 'Pad Thai, green curry, coconut milk, and herbs.',
    'American': 'Burgers, fries, BBQ ribs, and milkshakes.',
    'Chinese': 'Dim sum, stir-fry noodles, fried rice, and sweet sauces.'
}

def generate_restaurants(n=30):
    restaurants = []
    for i in range(n):
        cuisine = random.choice(cuisines)
        restaurants.append({
            'name': f'Restaurant {i+1}',
            'cuisine': cuisine,
            'description': descriptions[cuisine],
            'rating': round(random.uniform(3.0, 5.0), 2),
            'hours': random.choice(['11am - 10pm', '12pm - 11pm', '5pm - 2am'])
        })
    df = pd.DataFrame(restaurants)
    df.to_csv('data/restaurants.csv', index=False)
    print("✅ Generated data to data/restaurants.csv")

generate_restaurants()
