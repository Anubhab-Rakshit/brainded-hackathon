
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataLoader

def run_eda(data_path, output_dir):
    print("Loading data...")
    loader = DataLoader(data_path)
    loader.load_data()
    loader.preprocess()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Rating Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(loader.ratings['rating'], bins=10, kde=False)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/rating_distribution.png")
    plt.close()
    
    print("Generating Long Tail analysis...")
    # Calculate item popularity (number of ratings per movie)
    item_counts = loader.ratings['movieId'].value_counts().values
    plt.figure(figsize=(10, 6))
    plt.plot(item_counts)
    plt.title('Long Tail Distribution (Item Popularity)')
    plt.xlabel('Movie Rank')
    plt.ylabel('Number of Ratings')
    plt.yscale('log')
    plt.savefig(f"{output_dir}/long_tail.png")
    plt.close()
    
    print("Generating Genre Popularity...")
    # Explode genres
    movies_exploded = loader.movies.explode('genres_list')
    genre_counts = movies_exploded['genres_list'].value_counts()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Genre Popularity')
    plt.xlabel('Number of Movies')
    plt.savefig(f"{output_dir}/genre_popularity.png")
    plt.close()

    print(f"EDA Complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    DATA_PATH = "ml-latest-small"
    OUTPUT_DIR = "outputs/figures"
    run_eda(DATA_PATH, OUTPUT_DIR)
