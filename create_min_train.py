import pandas as pd

# Load the full train CSV
df = pd.read_csv('data/processed/20221204_amazon_reviews_train.csv')

# Get unique prod_ids
unique_prod_ids = df['prod_id'].unique()

# Select first 100 prod_ids
selected_prod_ids = unique_prod_ids[:100]

# Subsample: for each selected prod_id, take up to 8 reviews (random sample if more)
subsampled_df = pd.DataFrame()
for prod_id in selected_prod_ids:
    prod_reviews = df[df['prod_id'] == prod_id]
    if len(prod_reviews) > 8:
        prod_reviews = prod_reviews.sample(n=8, random_state=42)
    subsampled_df = pd.concat([subsampled_df, prod_reviews])

# Save to train.mini.csv
subsampled_df.to_csv('data/processed/train.mini.csv', index=False)

print(f"Created train.mini.csv with {len(subsampled_df)} rows from {len(selected_prod_ids)} products.")
