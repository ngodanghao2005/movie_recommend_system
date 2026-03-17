# 🎬 Movie Recommendation System

A machine learning-based movie recommendation system built on the MovieLens dataset, combining content-based filtering and regression modeling to predict user preferences and recommend movies.

# 🚀 Features

Key highlights of the system:

- 🎯 Content-based feature engineering (genres + genome tags)

- 👤 User profile construction using sparse matrix multiplication

- 🔗 Interaction-based feature modeling (user × movie)

- 📈 Regression-based rating prediction using SGDRegressor

- ⚡ Batch training for scalability on large datasets

- 💻 CLI-based recommendation interface

# 🧠 Tech Stack
| Category | Tools |
|----------|--------|
| Language | Python |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Sparse Matrix | SciPy |
# 📂 Project Structure

```
movie-recommendation-system/
│
├── data/                # Raw dataset (MovieLens)
│
├── models/              # Saved artifacts
│   ├── model.pkl
│   ├── movie_features.pkl
│   └── user_profiles.pkl
│
├── src/                 # Core modules
│   ├── build_features.py
│   ├── user_profile.py
│   └── recommend.py
│
├── train.py             # Training pipeline
├── recommend_cli.py     # CLI for recommendation
├── requirements.txt
└── README.md
```

# ⚙️ How to Run
## 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Train the model
```bash
python train.py
```

## 3️⃣ Get recommendations
```bash
python recommend_cli.py --user_id 10
```

# 📊 Model Overview
## 🔹 Input Features

- Movie feature vector (genres + tags)

- User profile vector

- Interaction features (element-wise multiplication)

## 🔹 Model

- SGDRegressor (Scikit-learn)

- Supports incremental learning (partial_fit) for large-scale data

# 📈 Evaluation Metrics

The model is evaluated using:

- 📉 Mean Absolute Error (MAE)

- 📉 Mean Squared Error (MSE)

- 📊 R² Score

# 🔮 Future Improvements

Planned enhancements:

- 📌 Add Precision@K and Recall@K

- 🎯 Implement Matrix Factorization (ALS, SVD)

- 🧠 Explore Neural Collaborative Filtering

- 🌐 Deploy as REST API using FastAPI

# 💡 Highlights

✔️ Combines content-based filtering + ML model

✔️ Uses real-world MovieLens dataset

✔️ Designed with modular, production-style structure

✔️ Ready for ML Engineer portfolio projects
