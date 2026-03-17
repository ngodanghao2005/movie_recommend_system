import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

def user_movie_profile(rating, movie_vector):
    rating = rating.copy()
    rating = rating[rating["movieId"].isin(movie_vector.index)]
    # scaler = MinMaxScaler()
    # weight = scaler.fit_transform(rating[["rating"]])
    # rating["rating"] = weight

    # Encode userId
    user_ids = rating["userId"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    rating["user_idx"] = rating["userId"].map(user_map)

    # Encode movieId
    movie_ids = movie_vector.index
    movie_map = {m: i for i, m in enumerate(movie_ids)}
    rating["movie_idx"] = rating["movieId"].map(movie_map)

    r = csr_matrix(
        (
            rating["rating"],
            (rating["user_idx"], rating["movie_idx"])
        ),
        shape=(len(user_ids), len(movie_ids))
    )

    m = movie_vector.values  # shape (num_movies, 119)
    user_profile = r.dot(m)
    user_counts = r.getnnz(axis=1).reshape(-1, 1)
    user_profile = user_profile / user_counts
    user_profile_vector = pd.DataFrame(
        user_profile,
        index=user_ids,
        columns=movie_vector.columns
    )
    user_profile_vector.index.name = "userId"

    return user_profile_vector