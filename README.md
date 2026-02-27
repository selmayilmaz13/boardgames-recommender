# BoardGame Recommender System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://boardgames-recommender.streamlit.app/)

**Live Demo:** [https://boardgames-recommender.streamlit.app/](https://boardgames-recommender.streamlit.app/)

## Project Overview

This project explores board game recommendation engines under conditions of extreme data sparsity. 

The dataset features:
- **1.17M+** user–game interactions
- **93K** users × **12K** games
- **98.98%** interaction sparsity
- **21K+** games with engineered categorical features (mechanics, themes, subcategories)

## Modeling Experiments

During the experimental modeling phase, several recommendation approaches were evaluated to address the sparsity challenge:

- **Collaborative Filtering:** Matrix factorization using Singular Value Decomposition (SVD, k=50) to capture latent user-item relationships.
- **Content-Based Filtering:** Utilizing cosine similarity over engineered categorical features to find games similar to a target game.
- **Hybrid Blending:** An ensemble approach combining Collaborative Filtering and Content-Based predictions (60% CF / 40% CB) to balance serendipity and relevance.
- **Maximal Marginal Relevance (MMR):** Applied to optimize the diversity of the final recommendation slate.

*Note: These models were evaluated for research and experimentation purposes to determine the optimal deployment strategy.*

## Deployed System

Due to hosting constraints and to ensure a highly responsive user experience, the deployed Streamlit application exclusively uses a **memory-efficient kNN-based content recommender**.

Key deployment optimizations:
- Replaced the $O(N^2)$ full similarity matrix with `sklearn.neighbors.NearestNeighbors` (`algorithm="brute", metric="cosine"`), significantly reducing the memory footprint for Streamlit Cloud.
- Dynamically loads and caches the fitted model and necessary datasets upon startup or user file upload.
- Employs dynamic dataset loading, gracefully falling back to user-uploaded files if bundled datasets are missing.

## Tech Stack

- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**
- **SciPy**
- **Streamlit**

## Run Locally

To run this application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/selmayilmaz13/boardgames-recommender.git
   cd boardgames-recommender
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   The repository includes a bundled dataset in the `data/` directory. 
   The deployed Streamlit application runs out-of-the-box with no additional setup required.


4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## Data & Attribution

- Dataset derived from BoardGameGeek via Kaggle: [threnjen/board-games-database-from-boardgamegeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek).
- Licensed under [Creative Commons Attribution-ShareAlike (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).
- Data was cleaned and preprocessed specifically for the development of these recommendation systems.
- **Powered by [BoardGameGeek](https://boardgamegeek.com)**.