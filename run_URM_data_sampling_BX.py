import pandas as pd
import numpy as np
import random

# Step 1: Load the Dataset
# Load the dataset from the provided CSV file
df = pd.read_csv('dataset_v2/BX/BX-Book-Ratings.csv', sep=';', header=0, names=['user_id', 'ISBN', 'Book-Rating'], encoding='ISO-8859-1')

# Step 2: Convert Rating Scale
# Convert the ratings from 0-10 scale to 1-5 scale and make them integers
df['Book-Rating'] = df['Book-Rating'].apply(lambda x: max(1, min(5, round(x / 2))) if x > 0 else 0)

# Step 3: Filter Users and Items with Sufficient Interactions
# Ensure all users have at least 10 items and all items have at least 10 users
df_filtered = df.copy()
while False:
    user_counts = df_filtered['user_id'].value_counts()
    item_counts = df_filtered['ISBN'].value_counts()
    
    # Filter out users with less than 10 interactions and items with less than 10 interactions
    df_filtered = df_filtered[df_filtered['user_id'].isin(user_counts[user_counts >= 10].index)]
    df_filtered = df_filtered[df_filtered['ISBN'].isin(item_counts[item_counts >= 10].index)]
    
    # Recalculate counts after filtering
    new_user_counts = df_filtered['user_id'].value_counts()
    new_item_counts = df_filtered['ISBN'].value_counts()
    
    # If no more users or items need to be filtered, break the loop
    if (new_user_counts >= 10).all() and (new_item_counts >= 10).all():
        break

# Step 4: Sampling Procedure
# Function to generate a sample from the original DataFrame
def generate_sample(df, min_users=100, min_items=100, avg_ratings_threshold=10):
    # Get unique users and items
    unique_users = df['user_id'].unique()
    unique_items = df['ISBN'].unique()
    
    # Randomly shuffle users and items
    random.shuffle(unique_users)
    random.shuffle(unique_items)
    
    # Randomly select the number of users and items
    num_users = random.randint(min_users, len(unique_users))
    num_items = random.randint(min_items, len(unique_items))
    
    # Select the users and items
    sampled_users = unique_users[:num_users]
    sampled_items = unique_items[:num_items]
    
    # Filter the original DataFrame to get interactions between the sampled users and items
    df_sample = df[(df['user_id'].isin(sampled_users)) & (df['ISBN'].isin(sampled_items))]
    
    # Check the average number of ratings per user
    avg_ratings_per_user = df_sample.groupby('user_id').size().mean()
    
    # If the sample meets the threshold, return it, otherwise generate a new one
    if avg_ratings_per_user >= avg_ratings_threshold:
        return df_sample
    else:
        return generate_sample(df, min_users, min_items, avg_ratings_threshold)

# Generate 600 samples
samples = []
for i in range(600):
    sample = generate_sample(df_filtered)
    samples.append(sample)
    print(f"Sample {i+1} generated with shape: {sample.shape}")

# Step 5: Save Samples for Further Analysis
# Saving the samples in the specified format with .inter extension
for idx, sample in enumerate(samples):
    filename = f'dataset_v2/BX/URM_subsets/sample_{idx + 1}.inter'
    sample.to_csv(filename, sep='	', header=['user_id', 'ISBN', 'Book-Rating'], index=False)

print("All samples generated and saved successfully.")
