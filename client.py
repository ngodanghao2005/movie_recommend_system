import argparse
import pandas as pd
import joblib

import src.recommend as re

parser = argparse.ArgumentParser(description="Movie Recommendation CLI")
parser.add_argument("--user_id", type=int, required=True)

args = parser.parse_args()
user_id = args.user_id

# Load data
movie = pd.read_csv("data/movie.csv")
rating = pd.read_csv("data/rating.csv")

model = joblib.load("models/model.pkl")
movie_vector_df = pd.read_pickle("models/movie_features.pkl")
user_profile_vector_df = pd.read_pickle("models/user_profiles.pkl")

# Recommend
list_id = re.recommend_system(
    user_id,
    rating,
    model,
    movie_vector_df,
    user_profile_vector_df
)

print(f"\n Recommendation for user {user_id}:\n")

for i in list_id:
    title = movie[movie["movieId"] == i]["title"].values[0]
    print("-", title)