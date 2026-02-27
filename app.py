import streamlit as st
import os
from recommender import load_data, build_feature_table, compute_similarity, recommend, explain_recommendations

st.set_page_config(page_title="Board Game Recommender", layout="wide")

@st.cache_data
def load_uploaded_tables(up_games, up_mech, up_themes, up_sub):
    import pandas as pd
    games = pd.read_csv(up_games)
    mechanics = pd.read_csv(up_mech)
    themes = pd.read_csv(up_themes)
    subcats = pd.read_csv(up_sub)

    for df in (games, mechanics, themes, subcats):
        df.columns = df.columns.str.strip().str.lower()

    return games, mechanics, themes, subcats

@st.cache_data
def get_games_tables():
    return load_data()

@st.cache_resource
def get_model_from_tables(games, mech, themes, sub):
    feats, bggids, X = build_feature_table(mech, themes, sub)
    nn_model = compute_similarity(X)
    return games, bggids, nn_model, feats

local_data_valid = False
try:
    local_data_exists = all(os.path.exists(f) for f in ["data/games.csv", "data/mechanics.csv", "data/themes.csv", "data/subcategories.csv"])
    if local_data_exists:
        # Pre-load to verify it's valid
        get_games_tables()
        local_data_valid = True
except Exception:
    local_data_valid = False

with st.sidebar:
    st.header("Dataset setup")
    
    use_uploads = False
    up_games = None
    up_mech = None
    up_themes = None
    up_sub = None
    
    if local_data_valid:
        st.success("Dataset loaded successfully (bundled version).")
    else:
        up_games = st.file_uploader("Upload games.csv", type="csv")
        up_mech = st.file_uploader("Upload mechanics.csv", type="csv")
        up_themes = st.file_uploader("Upload themes.csv", type="csv")
        up_sub = st.file_uploader("Upload subcategories.csv", type="csv")
        
        use_uploads = bool(up_games and up_mech and up_themes and up_sub)
        
        if use_uploads:
            st.info("Using uploaded dataset.")
        
    st.divider()

    st.header("Data & Attribution")
    st.write("Dataset source: BoardGameGeek via Kaggle dataset [threnjen/board-games-database-from-boardgamegeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek). Licensed under [Creative Commons Attribution-ShareAlike (CC BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/).")
    st.write("Dataset was cleaned and preprocessed for recommendation purposes.")
    st.write("Powered by [BoardGameGeek](https://boardgamegeek.com).")

if use_uploads:
    g_df, m_df, t_df, s_df = load_uploaded_tables(up_games, up_mech, up_themes, up_sub)
    games, bggids, nn_model, feats = get_model_from_tables(g_df, m_df, t_df, s_df)
elif local_data_valid:
    g_df, m_df, t_df, s_df = get_games_tables()
    games, bggids, nn_model, feats = get_model_from_tables(g_df, m_df, t_df, s_df)
else:
    st.error("Dataset not found! Please download the board-games-database-from-boardgamegeek dataset from Kaggle and upload the CSV files in the sidebar, or extract them into a `data/` folder locally.")
    st.info("Required files: `games.csv`, `mechanics.csv`, `themes.csv`, `subcategories.csv`")
    st.stop()

st.title("🎲 Board Game Recommender (Cosine Similarity)")
st.caption("Content-based recommender using cosine similarity over mechanics, themes, and subcategories. Explanations ranked by IDF importance.")
st.write("Select a game and get similar recommendations based on mechanics, themes, and subcategories.")

tab1, tab2 = st.tabs(["Similar Games", "Popular Baseline"])

with tab1:
    # Dropdown of game names
    game_list = games["name"].dropna().sort_values().unique().tolist()
    
    target_game = "Dungeons & Dragons Adventure Game"
    default_idx = game_list.index(target_game) if target_game in game_list else 0
    
    selected_game = st.selectbox("Choose a game", game_list, index=default_idx)

    top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    min_ratings = st.slider("Minimum number of user ratings", 0, 50000, 500, step=100)

    category_columns = [col for col in games.columns if col.startswith("cat:")]
    category_labels = [col.replace("cat:", "").title() for col in category_columns]
    selected_labels = st.multiselect("Filter by categories (optional)", category_labels)
    selected_categories = [f"cat:{label.lower()}" for label in selected_labels]

    year_range, weight_range, p_count, selected_playtime = None, None, None, None
    playtime_col = None

    if "yearpublished" in games.columns:
        import datetime
        current_year = datetime.datetime.now().year
        year_range = st.slider("YearPublished", min_value=1800, max_value=current_year, value=(1800, current_year))

    if "gameweight" in games.columns:
        weight_range = st.slider("GameWeight", min_value=0.0, max_value=5.0, value=(0.0, 5.0))

    if "minplayers" in games.columns and "maxplayers" in games.columns:
        p_count = st.slider("Players", min_value=1, max_value=12, value=4)

    if "mfgplaytime" in games.columns:
        playtime_col = "mfgplaytime"
    elif "commmaxplaytime" in games.columns:
        playtime_col = "commmaxplaytime"
        
    if playtime_col:
        selected_playtime = st.slider("Playtime", min_value=5, max_value=300, value=90)

    diversity = st.slider("Diversity (higher = more variety)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    st.caption("Filters apply after similarity ranking.")

    if st.button("Recommend"):
        recs = recommend(
            selected_game,
            games,
            bggids,
            nn_model,
            feats,
            top_n=top_n,
            min_user_ratings=min_ratings,
            required_categories=selected_categories,
            diversity_lambda=(1.0 - diversity)
        )
        
        if not recs.empty:
            cols_to_add = [c for c in ["yearpublished", "gameweight", "minplayers", "maxplayers", playtime_col] if c and c in games.columns and c not in recs.columns]
            if cols_to_add:
                recs = recs.merge(games[["bggid"] + cols_to_add], on="bggid", how="left")

            if year_range and "yearpublished" in recs.columns:
                recs = recs[(recs["yearpublished"] >= year_range[0]) & (recs["yearpublished"] <= year_range[1])]

            if weight_range and "gameweight" in recs.columns:
                recs = recs[(recs["gameweight"] >= weight_range[0]) & (recs["gameweight"] <= weight_range[1])]

            if p_count is not None and "minplayers" in recs.columns and "maxplayers" in recs.columns:
                recs = recs[(recs["minplayers"] <= p_count) & (recs["maxplayers"] >= p_count)]

            if playtime_col and selected_playtime is not None and playtime_col in recs.columns:
                recs = recs[(recs[playtime_col] <= selected_playtime)]

            if cols_to_add:
                recs = recs.drop(columns=cols_to_add)

        if recs.empty:
            st.warning("No results match your filters. Try widening filters or lowering minimum ratings.")
        else:
            # Add explain_recommendations column
            selected_bggid = int(games[games["name"] == selected_game].iloc[0]["bggid"])
            explanations = explain_recommendations(selected_bggid, recs["bggid"], feats, top_k=6)
            recs["Why Recommended"] = recs["bggid"].map(lambda x: " • ".join(explanations.get(x, [])))

            # Sorting and renaming
            recs = recs.sort_values(["similarity", "bayesavgrating"], ascending=[False, False])
            recs = recs.rename(columns={
                "bayesavgrating": "Bayes Avg Rating",
                "numuserratings": "# User Ratings"
            })

            # Reorder columns slightly for better view
            cols = ["name", "similarity", "Bayes Avg Rating", "# User Ratings", "Why Recommended"]
            st.dataframe(recs[cols], width="stretch")

with tab2:
    pop_top_k = st.slider("Top K", min_value=5, max_value=30, value=10)
    pop_min_ratings = st.slider("Minimum number of user ratings", min_value=0, max_value=50000, value=1000, step=100, key="pop_min_ratings")
    
    pop_df = games[games["numuserratings"] >= pop_min_ratings]
    pop_df = pop_df.sort_values("bayesavgrating", ascending=False).head(pop_top_k)
    pop_df = pop_df[["name", "bayesavgrating", "numuserratings", "yearpublished"]]
    pop_df = pop_df.rename(columns={
        "bayesavgrating": "Bayes Avg Rating",
        "numuserratings": "# User Ratings",
        "yearpublished": "Year Published"
    })
    st.dataframe(pop_df, width="stretch")

with st.expander("Dataset summary"):
    st.write(f"Games: {len(games):,}")
    st.write(f"Feature table rows: {len(feats):,} | Features: {feats.shape[1]-1:,}")
    st.dataframe(games.head(10), width="stretch")