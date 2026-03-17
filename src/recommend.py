import numpy as np

def recommend_system(userid_rec, rating, model, movie_vector_df, user_profile_vector_df):
    if userid_rec not in user_profile_vector_df.index:
        return []
    rated_movies = rating[rating["userId"] == userid_rec]["movieId"]

    candidate_movies = movie_vector_df.loc[
        ~movie_vector_df.index.isin(rated_movies)
    ]
    user_vec = user_profile_vector_df.loc[userid_rec].values
    user_batch = np.tile(user_vec, (len(candidate_movies), 1))
    interaction = candidate_movies.values * user_batch

    input_data = np.hstack([candidate_movies.values, user_batch, interaction])
    output_data = model.predict(input_data)

    list_movie = sorted(zip(candidate_movies.index, output_data),
                        key=lambda x: x[1], reverse=True)[:5]

    list_movie_id = [x[0] for x in list_movie]
    return list_movie_id