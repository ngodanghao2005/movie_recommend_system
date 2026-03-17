📌 Movie Recommendation System

A machine learning-based movie recommendation system built using the MovieLens dataset. This project combines content-based filtering and regression modeling to predict user preferences and recommend movies.

🚀 Features

Content-based feature engineering (genres + genome tags)

User profile construction using sparse matrix

Interaction-based feature modeling

Regression-based rating prediction (SGDRegressor)

Batch training for scalability

CLI-based recommendation system

🧠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

SciPy

📂 Project Structure
movie-recommendation-system/
│
├── data/                # Dataset
├── models/              # Saved models
├── src/                 # Core modules
│   ├── build_features.py
│   ├── user_profile.py
│   └── recommend.py
│
├── train.py             # Train model
├── recommend_cli.py     # Run recommendation
├── requirements.txt
└── README.md
⚙️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train model
python train.py
3️⃣ Get recommendation
python recommend_cli.py --user_id 10
📊 Model

Input features:

Movie features

User profile

Interaction (element-wise multiplication)

Model:

SGDRegressor

📈 Evaluation Metrics

MAE

MSE

R² Score

🔮 Future Improvements

Add Precision@K, Recall@K

Matrix Factorization (ALS, SVD)

Neural Collaborative Filtering

Deploy as API (FastAPI)
