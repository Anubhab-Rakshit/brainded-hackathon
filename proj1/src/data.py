import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.movies = None
        self.ratings = None
        self.tags = None
        self.links = None

    def load_data(self):
        """Loads data from CSV files."""
        self.movies = pd.read_csv(f"{self.data_path}/movies.csv")
        self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv")
        self.tags = pd.read_csv(f"{self.data_path}/tags.csv")
        self.links = pd.read_csv(f"{self.data_path}/links.csv")

    def preprocess(self):
        """Preprocesses the data: timestamps, genres."""
        if self.ratings is not None:
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        
        if self.tags is not None:
            self.tags['timestamp'] = pd.to_datetime(self.tags['timestamp'], unit='s')

        if self.movies is not None:
            # Genres are pipe-separated
            self.movies['genres_list'] = self.movies['genres'].apply(lambda x: x.split('|'))

    def get_train_test_split(self, method='leave_last_n', n=1, min_ratings=5):
        """
        Splits data into train and test sets.
        
        Args:
            method: 'leave_last_n' or 'time_split'
            n: Number of items to leave for test per user (for leave_last_n)
            min_ratings: Minimum ratings a user must have to be included in test split
            
        Returns:
            train_df, test_df
        """
        if self.ratings is None:
            raise ValueError("Ratings data not loaded yet.")

        # Filter users with enough ratings
        user_counts = self.ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        filtered_ratings = self.ratings[self.ratings['userId'].isin(valid_users)].copy()
        
        # Sort by user and timestamp
        filtered_ratings = filtered_ratings.sort_values(['userId', 'timestamp'])

        if method == 'leave_last_n':
            # Create a rank for each rating per user (descending order of time)
            # method='first' in rank ensures unique ranks
            filtered_ratings['rank'] = filtered_ratings.groupby('userId')['timestamp'].rank(method='first', ascending=False)
            
            test_mask = filtered_ratings['rank'] <= n
            test_df = filtered_ratings[test_mask].drop(columns=['rank'])
            train_df = filtered_ratings[~test_mask].drop(columns=['rank'])
            
            return train_df, test_df
        
        else:
            raise NotImplementedError(f"Split method {method} not implemented.")

    def get_full_interaction_matrix(self):
        """Returns the user-item interaction matrix (pivot table)."""
        if self.ratings is None:
            self.load_data()
        
        return self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
