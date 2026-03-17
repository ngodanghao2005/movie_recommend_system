import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def build_movie_features(movie, genome_scores, genome_tags, top_k_tags=100):
    movie["genres"] = movie["genres"].str.split("|")
    movie = movie[movie["genres"].apply(lambda x: "(no genres listed)" not in x)]

    # Filter movie theo genome_scores
    genome_ids = set(genome_scores["movieId"])
    movie = movie[movie["movieId"].isin(genome_ids)]

    mlb = MultiLabelBinarizer()
    multi_hot = mlb.fit_transform(movie["genres"])
    multi_hot_df = pd.DataFrame(multi_hot, columns=mlb.classes_, index=movie["movieId"])

    # 4️⃣ Select top K tag
    grouped_rele = genome_scores.groupby(["tagId"])["relevance"]
    list_rele = grouped_rele.mean().sort_values(ascending=False)[:top_k_tags]

    list_rele = list(list_rele.index)

    genome_top = genome_scores[genome_scores["tagId"].isin(list_rele)]
    genome_top = genome_top.merge(
        genome_tags[["tagId", "tag"]],
        on="tagId",
        how="left"
    )

    rele_matrix = genome_top.pivot_table(index="movieId", columns="tag", values="relevance")
    movie_vector = pd.concat([multi_hot_df, rele_matrix], axis=1, join="inner").astype("float32")

    return movie_vector