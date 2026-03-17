import pandas as pd
import numpy as np
import joblib
import src.build_features as bf, src.user_profile as up

from sklearn.model_selection import train_test_split
# from ydata_profiling import ProfileReport
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

genome_scores = pd.read_csv("data/genome_scores.csv")
# gscore_profile = ProfileReport(genome_scores, title="Genome_Score Profile")
# gscore_profile.to_file("gscore_report.html")

genome_tags = pd.read_csv("data/genome_tags.csv")

movie = pd.read_csv("data/movie.csv")
# movie_profile = ProfileReport(movie, title="Movie Profile")
# movie_profile.to_file("movie_report.html")

rating = pd.read_csv("data/rating.csv")

movie_vector_df = bf.build_movie_features(movie, genome_scores, genome_tags)
train_rating, test_rating = train_test_split(rating, test_size=0.2, random_state=42)
user_profile_vector_df = up.user_movie_profile(train_rating, movie_vector_df)

movie_vector_df = movie_vector_df.add_prefix("movie_")
user_profile_vector_df = user_profile_vector_df.add_prefix("user_")

# Training
model = SGDRegressor()
batch_size = 10000
for i in range(0, len(train_rating), batch_size):
    batch = train_rating.iloc[i : i + batch_size]
    batch = batch[
        batch["movieId"].isin(movie_vector_df.index) &
        batch["userId"].isin(user_profile_vector_df.index)
        ]

    movie_batch = movie_vector_df.loc[batch["movieId"]]
    user_batch = user_profile_vector_df.loc[batch["userId"]]
    interaction = movie_batch.values * user_batch.values

    x_train = np.hstack([movie_batch.values, user_batch.values, interaction])
    y_train = batch["rating"].values
    model.partial_fit(x_train, y_train)

# Evaluating
mae_list, mse_list, r2_list = [], [], []
for i in range(0, len(test_rating), batch_size):
    batch = test_rating.iloc[i:i + batch_size]
    batch = batch[
        batch["movieId"].isin(movie_vector_df.index) &
        batch["userId"].isin(user_profile_vector_df.index)
    ]

    movie_batch = movie_vector_df.loc[batch["movieId"]]
    user_batch = user_profile_vector_df.loc[batch["userId"]]
    interaction = movie_batch.values * user_batch.values

    x_test = np.hstack([movie_batch.values, user_batch.values, interaction])
    y_test = batch["rating"].values
    y_predict = model.predict(x_test)
    y_predict = np.clip(y_predict, 0.5, 5)

    mae_list.append(mean_absolute_error(y_test, y_predict))
    mse_list.append(mean_squared_error(y_test, y_predict))
    r2_list.append(r2_score(y_test, y_predict))

print("Average MAE: {}".format(np.average(mae_list)))
print("Average MSE: {}".format(np.average(mse_list)))
print("Average R2 score: {}".format(np.average(r2_list)))

joblib.dump(model, "models/model.pkl")
movie_vector_df.to_pickle("models/movie_features.pkl")
user_profile_vector_df.to_pickle("models/user_profiles.pkl")

print("Model saved successfully!")