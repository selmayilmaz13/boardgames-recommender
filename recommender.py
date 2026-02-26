import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

def load_data():
    games = pd.read_csv(f"{DATA_DIR}/games.csv")
    mechanics = pd.read_csv(f"{DATA_DIR}/mechanics.csv")
    themes = pd.read_csv(f"{DATA_DIR}/themes.csv")
    subcats = pd.read_csv(f"{DATA_DIR}/subcategories.csv")

    # Standardize column names
    for df in (games, mechanics, themes, subcats):
        df.columns = df.columns.str.strip().str.lower()

    return games, mechanics, themes, subcats


def build_feature_table(mechanics, themes, subcats):
    """
    Merge mechanics, themes, and subcategories
    to create one feature table per game (bggid).
    """
    feats = mechanics.merge(themes, on="bggid", how="inner") \
                     .merge(subcats, on="bggid", how="inner")

    feats = feats.fillna(0)

    bggids = feats["bggid"].to_numpy()
    X = feats.drop(columns=["bggid"]).to_numpy(dtype=float)

    return feats, bggids, X


def compute_similarity(X):
    return cosine_similarity(X)


def recommend(game_name, games, bggids, sim, top_n=10, min_user_ratings=200, required_categories=None, diversity_lambda=1.0):
    if required_categories is None:
        required_categories = []
    """
    Recommend similar games based on cosine similarity.
    Filters out games with very few ratings.
    """
    match = games[games["name"].astype(str).str.lower() == game_name.lower()]
    if match.empty:
        return pd.DataFrame()

    target_id = int(match.iloc[0]["bggid"])
    idx_arr = np.where(bggids == target_id)[0]
    if len(idx_arr) == 0:
        return pd.DataFrame()

    idx = idx_arr[0]
    scores = sim[idx].copy()
    scores[idx] = -1  # exclude itself

    # take more than top_n first, then filter by min_user_ratings
    candidate_multiplier = 50
    top_idx = np.argsort(scores)[::-1][: top_n * candidate_multiplier]

    rec_ids = bggids[top_idx]
    rec_scores = scores[top_idx]

    results = games[games["bggid"].isin(rec_ids)].copy()

    # add similarity
    score_map = {int(i): float(s) for i, s in zip(rec_ids, rec_scores)}
    results["similarity"] = results["bggid"].map(score_map)

    # filter by minimum user ratings
    results = results[results["numuserratings"].fillna(0) >= min_user_ratings]

    # filter by required categories
    if required_categories:
        for cat in required_categories:
            if cat in results.columns:
                results = results[results[cat] == 1]

    # Apply MMR (Maximal Marginal Relevance) before returning
    # We want to iteratively pick up to top_n candidates
    if not results.empty and top_n > 0:
        candidate_bool = bggids.reshape(-1, 1) == results["bggid"].to_numpy() # shape: (len(bggids), len(results))
        candidate_indices = np.where(candidate_bool.any(axis=1))[0]
        
        # Build mapping from bggid to sim index
        bggid_to_idx = {bggids[i]: i for i in candidate_indices}
        
        selected_bggids = []
        candidate_bggids = results["bggid"].tolist()
        
        while len(selected_bggids) < top_n and candidate_bggids:
            best_score = -np.inf
            best_cand = None
            
            for cand in candidate_bggids:
                cand_idx = bggid_to_idx[cand]
                sim_to_query = sim[idx, cand_idx]
                
                if not selected_bggids:
                    score = sim_to_query
                else:
                    # Max sim between this candidate and any already selected candidate
                    selected_indices = [bggid_to_idx[s] for s in selected_bggids]
                    max_sim_to_selected = np.max(sim[cand_idx, selected_indices])
                    
                    # MMR formula
                    score = (diversity_lambda * sim_to_query) - ((1.0 - diversity_lambda) * max_sim_to_selected)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            selected_bggids.append(best_cand)
            candidate_bggids.remove(best_cand)
            
        # Re-sort results DataFrame based on MMR selection order
        results = results.set_index("bggid").loc[selected_bggids].reset_index()

    # keep clean columns
    results = results[["bggid", "name", "bayesavgrating", "numuserratings", "similarity"]].copy()

    return results

def clean_feature_name(feat):
    """
    Clean feature name: replace underscores/multiple spaces with single space.
    Title-case, but keep known acronyms like TV, RPG uppercase.
    """
    import re
    # clean out cat: prefix if any
    feat = feat.replace("cat:", "")
    # replace underscores and multiple spaces
    feat = re.sub(r'[_\s]+', ' ', feat).strip()
    
    # Title case words but preserve specific uppercase acronyms
    keep_upper = {"TV", "RPG", "CCG", "TCG", "LCG", "WWII", "WWI", "IP"}
    words = feat.split()
    cleaned_words = []
    for w in words:
        if w.upper() in keep_upper:
            cleaned_words.append(w.upper())
        else:
            cleaned_words.append(w.capitalize())
            
    return " ".join(cleaned_words)

def explain_recommendations(selected_bggid, rec_bggids, feats_df, top_k=8):
    """
    Returns a dict {bggid: [shared_feature_names...]} explaining why
    each rec_bggid was recommended based on shared binary features,
    ranked by descending IDF (rarer features first).
    """
    explanations = {}
    
    # Get the feature row for the selected game
    selected_row = feats_df[feats_df["bggid"] == selected_bggid]
    if selected_row.empty:
        return {b: [] for b in rec_bggids}
        
    feature_cols = feats_df.drop(columns=["bggid"])
    
    # Compute IDF per feature (exclude bggid)
    # df = number of games with feature == 1
    doc_freq = (feature_cols == 1).sum()
    N = len(feats_df)
    idf_scores = np.log((N + 1) / (doc_freq + 1)) + 1
    
    selected_features = selected_row.iloc[0].drop("bggid")
    selected_active = selected_features[selected_features == 1].index

    for rec_id in rec_bggids:
        rec_row = feats_df[feats_df["bggid"] == rec_id]
        if rec_row.empty:
            explanations[rec_id] = []
            continue
            
        rec_features = rec_row.iloc[0].drop("bggid")
        rec_active = rec_features[rec_features == 1].index
        
        # Intersection of active features
        shared = set(selected_active).intersection(set(rec_active))
        
        # Sort by IDF score descending
        shared_ranked = sorted(list(shared), key=lambda x: idf_scores[x], reverse=True)[:top_k]
        
        # Clean labels
        cleaned_ranked = [clean_feature_name(f) for f in shared_ranked]
        explanations[rec_id] = cleaned_ranked
        
    return explanations