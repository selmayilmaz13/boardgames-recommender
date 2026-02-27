import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)
    return nn


def recommend(game_name, games, bggids, nn_model, feats, top_n=10, min_user_ratings=200, required_categories=None, diversity_lambda=1.0):
    if required_categories is None:
        required_categories = []
    """
    Recommend similar games based on cosine similarity logic without a full matrix.
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
    
    # Get feature vector for target
    target_vec = feats[feats["bggid"] == target_id].drop(columns=["bggid"]).to_numpy(dtype=float)

    # Query nn_model
    candidate_multiplier = 50
    n_neighbors_to_query = min(top_n * candidate_multiplier, len(bggids))
    distances, indices = nn_model.kneighbors(target_vec, n_neighbors=n_neighbors_to_query)
    
    # Flatten arrays
    distances = distances[0]
    indices = indices[0]
    
    # Exclude the target itself from the results
    mask = indices != idx
    indices = indices[mask]
    distances = distances[mask]
    
    rec_scores = 1.0 - distances
    rec_ids = bggids[indices]

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
        candidate_bggids = results["bggid"].tolist()
        
        cand_feats_df = feats.set_index("bggid").loc[candidate_bggids]
        cand_vecs = cand_feats_df.to_numpy(dtype=float)
        cand_to_idx = {cand: i for i, cand in enumerate(candidate_bggids)}
        
        selected_bggids = []
        selected_indices = []
        
        while len(selected_bggids) < top_n and candidate_bggids:
            best_score = -np.inf
            best_cand = None
            best_cand_idx = None
            
            for cand in candidate_bggids:
                cand_idx = cand_to_idx[cand]
                sim_to_query = score_map.get(int(cand), 0.0)
                
                if not selected_bggids:
                    score = sim_to_query
                else:
                    cand_vec = cand_vecs[cand_idx:cand_idx+1]
                    sel_vecs = cand_vecs[selected_indices]
                    sims = cosine_similarity(cand_vec, sel_vecs)
                    max_sim_to_selected = np.max(sims)
                    
                    # MMR formula
                    score = (diversity_lambda * sim_to_query) - ((1.0 - diversity_lambda) * max_sim_to_selected)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_cand_idx = cand_idx
            
            selected_bggids.append(best_cand)
            selected_indices.append(best_cand_idx)
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