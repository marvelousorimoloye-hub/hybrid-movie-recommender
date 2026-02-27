import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD
from scipy import sparse
# ─── Load data & models ───
@st.cache_data
def load_data():

    movies = pd.read_csv('data/ml-latest-small/processed/movies_fully_enriched_with_content_text.csv')
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    
    return movies, ratings

@st.cache_resource
def load_models():
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split

    # Load ratings (already loaded in load_data())
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()  # use all data for demo

    algo = SVD(n_factors=150, n_epochs=40, lr_all=0.001, reg_all=0.08, random_state=42)
    algo.fit(trainset)
    return algo

@st.cache_resource
def get_cosine_similarity():
    tfidf_matrix_genre_boosted = sparse.load_npz('data/ml-latest-small/processed/tfidf_matrix_genre_boosted.npz')
    st.info("Computing cosine similarity matrix (happens once)...")
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(tfidf_matrix_genre_boosted)  
    return cosine_sim

movies, ratings = load_data()
loaded_algo = load_models()
cosine_sim = get_cosine_similarity()

# Title → index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()




def hybrid_recommend_vectorized(
    user_id=None,
    title=None,                 # optional anchor movie title
    n=10,
    alpha=0.45,
    min_ratings_threshold=15,   # below this → cold user
    max_candidates=5000,   # limit candidates to speed up
    fallback_to_popularity=True
):
    """
    Fast vectorized hybrid recommender with cold-start handling.
    
    Cold-start logic:
    - If user has < min_ratings_threshold ratings → cold user
      - If title provided → pure content-based similarity
      - Else → popularity-based fallback
    - Otherwise → hybrid (content + collaborative)
    
    Uses vectorized operations on candidate subset → much faster.
    """
    # ─── Pre-filter candidates ───
    # Only consider reasonably popular movies to keep computation fast
    popular = movies.sort_values('rating_count', ascending=False).head(max_candidates)
    all_candidates = popular['movieId'].values
    
        
    if user_id is not None:
        user_rated = ratings[ratings['userId'] == user_id]['movieId'].values
        user_rating_count = len(user_rated)
        candidates = np.setdiff1d(all_candidates, user_rated)
    else:
        user_rated = np.array([])
        user_rating_count = 0
        candidates = all_candidates

    is_cold_user = user_rating_count < min_ratings_threshold

    # ─── Cold-start path ───
    if is_cold_user:
        print(f"User {user_id} is cold ({user_rating_count} ratings) → fallback mode")

        if title is not None and title in indices:
            # Pure content-based from anchor movie
            print(f"Using content-based similarity to '{title}'")
            idx = indices[title]
            candidate_indices = movies[movies['movieId'].isin(candidates)].index.values
            anchor_movie_id = movies[movies['title'] == title]['movieId'].iloc[0]
            
            sim_values = cosine_sim[idx, candidate_indices]
            if issparse(sim_values):
                sim_values = sim_values.toarray().flatten()
            else:
                sim_values = sim_values.flatten()
           
            temp_df = pd.DataFrame({
            'index': candidate_indices,
            'sim': sim_values,
            'movieId': movies.iloc[candidate_indices]['movieId'].values
            })
    
            # Exclude the anchor movie
            temp_df = temp_df[temp_df['movieId'] != anchor_movie_id]
    
            # Sort and take top N
            top = temp_df.sort_values('sim', ascending=False).head(n)
    
            recs = movies.iloc[top['index']][['title', 'genres']].copy()
                
            recs['hybrid_score'] = top['sim'].values * 5  # 0–5 scale
    
            return recs.reset_index(drop=True)
    
           
        elif fallback_to_popularity:
            # Most popular movies
            print("No anchor → returning most popular movies")
            popular_recs = popular[['title', 'genres']].head(n).copy()
            popular_recs['hybrid_score'] = np.nan
            return popular_recs

        else:
            return pd.DataFrame()  # empty

    # ─── Normal hybrid path (warm user) ───
    candidate_indices = movies[movies['movieId'].isin(candidates)].index.values

    # 1. Collaborative scores (vectorized predict is not native, so still loop but only on 2000)
    collab_scores = np.array([
        loaded_algo.predict(user_id, mid).est for mid in candidates
    ])
    
    # 2. Content similarity scores (vectorized)
    if title is not None:
    # Anchor on one movie
        anchor_idx = indices[title]
        content_sims = cosine_sim[anchor_idx, candidate_indices]
        content_sims = content_sims.toarray().ravel() if issparse(content_sims) else np.ravel(content_sims)
       
    else:
    # Average similarity to user's liked movies
        user_liked = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4.0)]['movieId']
        if len(user_liked) == 0:
            content_sims = np.zeros(len(candidates))
        else:
            liked_indices = movies[movies['movieId'].isin(user_liked)].index.values
            sim_matrix = cosine_sim[liked_indices[:, None], candidate_indices]
             # Force mean to 1D
            content_sims = sim_matrix.mean(axis=0)
            content_sims = content_sims.toarray().ravel() if issparse(content_sims) else np.ravel(content_sims)

    # Normalize to 0–5 scale
    content_scores = content_sims * 5.0

    # Collaborative scores (still loop, but limited to max_candidates)
    collab_scores = np.array([
    loaded_algo.predict(user_id, mid).est for mid in candidates
    ])

    # Hybrid score
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores

    # Create DataFrame
    results = pd.DataFrame({
    'movieId': candidates,
    'hybrid_score': hybrid_scores
    })
   
    # 2. Content similarity scores (vectorized)
   
    results = results.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    results = results.sort_values('hybrid_score', ascending=False).head(n)

    return results[['title', 'genres', 'hybrid_score']]

# ─── Streamlit App ───
st.title("Movie Recommender System")
st.markdown("Hybrid recommender: content + collaborative filtering")

# Sidebar controls
st.sidebar.header("Settings")
alpha = st.sidebar.slider("Content weight (α)", 0.0, 1.0, 0.45, 0.05)
n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# User input
# ─── User input section ───
st.subheader("Your information")

col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input(
        "User ID (for personalized recommendations)",
        min_value=1,
        max_value=610,
        value=414,
        step=1,
        help="Leave at default or enter any valid user ID from MovieLens"
    )

with col2:
    anchor_title = st.selectbox(
        "Movie you just watched / liked (optional)",
        options=["(none – use only user profile)"] + sorted(movies['title'].unique().tolist()),
        index=0,
        help="If you provide a movie, recommendations will be influenced by similarity to it"
    )

# Convert "(none)" back to None
if anchor_title == "(none – use only user profile)":
    anchor_title = None
# Generate button
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        try:
            recs = hybrid_recommend_vectorized(
                user_id=user_id if user_id > 0 else None,
                title=anchor_title,
                n=n_recs,
                alpha=alpha,
               
            )

            if recs.empty:
                st.warning("No recommendations available. Try providing more information.")
            else:
                st.success(f"Top {len(recs)} recommendations")
                st.dataframe(recs.style.format({'hybrid_score': '{:.2f}'}))

                # Posters (same as before)
                st.subheader("Posters")
                cols = st.columns(min(5, len(recs)))
                for i, row in recs.iterrows():
                    movie = movies[movies['title'] == row['title']].iloc[0]
                    poster_path = movie.get('poster_path')
                    if pd.notna(poster_path) and poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                        with cols[i % 5]:
                            st.image(poster_url, width=120, caption=row['title'])
                    else:
                        with cols[i % 5]:
                            st.write(f"No poster available")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Check if models/data are loaded correctly.")

# Footer
st.markdown("---")
st.caption("Built with MovieLens ml-latest-small + TMDB enrichment | Hybrid SVD + Content TF-IDF")