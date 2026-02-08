
import sys
import os
# Fix for PyTorch MPS 'aten::_nested_tensor_from_mask_left_aligned' crash
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import DataLoader
from src.recommenders import PopularityRecommender, CollaborativeRecommender, SVDRecommender, ContentRecommender, HybridRecommender, NeuralCFRecommender, DiversityRecommender, LightGCNRecommender, EnsembleRecommender, SASRecRecommender
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, coverage
from src.explainability import Explainer

def evaluate_model(model, train_df, test_df, all_items, k=10):
    # model.fit(train_df) - Moved outside to handle differing signatures
    
    # Test users
    test_users = test_df['userId'].unique()
    
    precisions = []
    recalls = []
    ndcgs = []
    all_recs = []
    
    for uid in tqdm(test_users, desc="Evaluating"):
        # Ground truth
        truth = test_df[test_df['userId'] == uid]['movieId'].tolist()
        
        # Prediction
        recs = model.recommend(uid, n=k)
        all_recs.append(recs)
        
        precisions.append(precision_at_k(recs, truth, k))
        recalls.append(recall_at_k(recs, truth, k))
        ndcgs.append(ndcg_at_k(recs, truth, k))
        
    metrics = {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'NDCG@K': np.mean(ndcgs),
        'Coverage': coverage(all_recs, all_items)
    }
    return metrics, all_recs

def main():
    print("Initializing ReelSense Pipeline...")
    data_path = "ml-latest-small"
    loader = DataLoader(data_path)
    loader.load_data()
    loader.preprocess()
    
    print("Splitting Data (Leave-Last-1)...")
    train_df, test_df = loader.get_train_test_split(method='leave_last_n', n=1)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    all_items = loader.movies['movieId'].unique()
    
    models = {
        "Popularity": PopularityRecommender(),
        "User-User CF": CollaborativeRecommender(method='user_user', n_neighbors=50),
        "SVD": SVDRecommender(n_components=20),
        "Hybrid": HybridRecommender(SVDRecommender(n_components=20), ContentRecommender(), alpha=0.3),
        "Neural CF (DL)": NeuralCFRecommender(embedding_dim=32, n_epochs=10),
        "LightGCN (SOTA)": LightGCNRecommender(n_epochs=50),
        "SASRec (Transformer)": SASRecRecommender(n_epochs=50, embedding_dim=64, n_heads=2),
        "Hybrid + Diversity (MMR)": DiversityRecommender(HybridRecommender(SVDRecommender(n_components=20), ContentRecommender(), alpha=0.3), lambda_param=0.6)
    }
    
    # Define Ensemble separately to reference processed models
    # Note: In this pipeline script, we execute sequentially. 
    # To properly ensemble, we'd need the trained instances.
    # We will add it to the 'models' dict but we need to ensure the inner models are trained.
    
    # The "Holy Trinity" Ensemble
    # SVD (Static Global) + LightGCN (Graph Relational) + SASRec (Temporal Sequential)
    
    svd_ref = models["SVD"]
    gcn_ref = models["LightGCN (SOTA)"]
    sas_ref = models["SASRec (Transformer)"]
    
    models["Ensemble (Trinity)"] = EnsembleRecommender({
        "SVD": svd_ref,
        "LightGCN": gcn_ref,
        "SASRec": sas_ref
    }, weights={"SVD": 0.3, "LightGCN": 0.3, "SASRec": 0.4}) # Slightly higher weight to the "smartest" model
    
    results = {}
    recommendations = {}
    
    for name, model in models.items():
        print(f"\nTraining and Evaluating {name}...")
        
        if name == "Hybrid" or name == "Hybrid + Diversity (MMR)":
            if "MMR" in name:
                # DiversityRecommender.fit handles kwargs for base model
                model.fit(train_df, loader.movies, tags_df=loader.tags)
            else:
                model.fit(train_df, loader.movies, loader.tags)
        elif name == "SASRec (Transformer)":
            # Pass movies for Genre Embeddings
            model.fit(train_df, movies_df=loader.movies)
        elif "Ensemble" in name:
            print("Ensemble uses pre-trained models. Skipping fit.")
        else:
            model.fit(train_df)
            
        metrics, recs = evaluate_model(model, train_df, test_df, all_items, k=10)
        results[name] = metrics
        recommendations[name] = recs
        print(f"{name} Results: {metrics}")
        
    # Explainability Demo
    print("\n--- Explainability Demo ---")
    explainer = Explainer(loader, train_df)
    
    # Pick a sample user from test set
    sample_user = test_df['userId'].iloc[0]
    
    # Get recommendations from SVD
    svd_model = models['SVD']
    recs = svd_model.recommend(sample_user, n=3)
    
    print(f"User ID: {sample_user}")
    print(f"Profile: {explainer.user_profiles.get(sample_user, 'Unknown')}")
    
    for movie_id in recs:
        movie_title = loader.movies[loader.movies['movieId'] == movie_id]['title'].iloc[0]
        explanation = explainer.explain(sample_user, movie_id)
        print(f"Recommended: {movie_title}")
        print(f"Explanation: {explanation}")
        
    # Formatting Final Report
    results_df = pd.DataFrame(results).T
    results_df = results_df[['Precision@K', 'Recall@K', 'NDCG@K', 'Coverage']]
    print("\n\n--- Final Evaluation Championship ---")
    print(results_df.sort_values(by='NDCG@K', ascending=False))
    
    winner = results_df['NDCG@K'].idxmax()
    print(f"\n🏆 CHAMPION MODEL: {winner} 🏆")
    print("Reason: Best balance of ranking quality (NDCG) and diversity.")
    print(results_df)

if __name__ == "__main__":
    main()
