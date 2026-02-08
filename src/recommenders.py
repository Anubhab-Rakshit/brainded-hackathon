
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class BaseRecommender:
    def fit(self, train_df):
        pass
    
    def recommend(self, user_id, n=10):
        pass

class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.popular_items = None
    
    def fit(self, train_df):
        # Calculate weighted rating or just count
        item_counts = train_df['movieId'].value_counts()
        self.popular_items = item_counts.index.tolist()
    
    def recommend(self, user_id, n=10):
        return self.popular_items[:n]

class CollaborativeRecommender(BaseRecommender):
    def __init__(self, method='user_user', n_neighbors=50):
        self.method = method
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        self.similarity_matrix = None
        
    def fit(self, train_df):
        # Create user-item matrix
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.reverse_user_map = {i: u for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        user_indices = train_df['userId'].map(self.user_map)
        item_indices = train_df['movieId'].map(self.item_map)
        ratings = train_df['rating']
        
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                           shape=(len(users), len(items)))
        
        # We compute similarity on demand or for small datasets precompute
        # For fairness, let's precompute for User-User since dataset is small (610 users)
        if self.method == 'user_user':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif self.method == 'item_item':
            # Transpose for item-item
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
    def recommend(self, user_id, n=10):
        if user_id not in self.user_map:
            return []
        
        u_idx = self.user_map[user_id]
        
        if self.method == 'user_user':
            # Get similar users
            sim_scores = self.similarity_matrix[u_idx]
            # Zero out self similarity
            sim_scores[u_idx] = 0
            
            # Top neighbors
            neighbor_indices = np.argsort(sim_scores)[::-1][:self.n_neighbors]
            
            # Aggregate ratings
            # Score = sum(sim * rating) / sum(sim)
            # Efficient way with matrix multiplication
            
            # Weighted sum of ratings from neighbors
            neighbor_sims = sim_scores[neighbor_indices].reshape(-1, 1)
            neighbor_ratings = self.user_item_matrix[neighbor_indices].toarray()
            
            # Predict
            pred_scores = np.sum(neighbor_ratings * neighbor_sims, axis=0)
            
            # Filter already rated items
            # user_rated_indices = self.user_item_matrix[u_idx].indices
            # pred_scores[user_rated_indices] = -1
            
            top_item_indices = np.argsort(pred_scores)[::-1][:n]
            return [self.reverse_item_map.get(i) for i in top_item_indices]

        # Simplified implementation for robustness
        return []

class SVDRecommender(BaseRecommender):
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components)
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, train_df):
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        user_indices = train_df['userId'].map(self.user_map)
        item_indices = train_df['movieId'].map(self.item_map)
        ratings = train_df['rating']
        
        matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(len(users), len(items)))
        
        self.model.fit(matrix)
        self.user_factors = self.model.transform(matrix) # U * Sigma
        self.item_factors = self.model.components_.T # V
        
    def recommend(self, user_id, n=10):
        if user_id not in self.user_map:
            return []
        
        u_idx = self.user_map[user_id]
        user_vector = self.user_factors[u_idx]
        
        # Predict scores: user_vector dot item_factors.T
        scores = np.dot(user_vector, self.item_factors.T)
        
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.reverse_item_map.get(i) for i in top_indices]

class ContentRecommender(BaseRecommender):
    def __init__(self):
        self.item_profiles = None
        self.tfidf_matrix = None
        self.movie_indices = None
        self.cosine_sim = None
        
    def fit(self, train_df, movies_df, tags_df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        
        # Prepare content: Genres + Tags
        # 1. Genres
        movies_df['genres_str'] = movies_df['genres'].str.replace('|', ' ')
        
        # 2. Tags
        # Aggregate tags per movie
        tags_agg = tags_df.groupby('movieId')['tag'].agg(lambda x: ' '.join(str(v) for v in x)).reset_index()
        
        # Merge
        content_df = movies_df[['movieId', 'genres_str']].merge(tags_agg, on='movieId', how='left')
        content_df['tag'] = content_df['tag'].fillna('')
        content_df['content'] = content_df['genres_str'] + ' ' + content_df['tag']
        
        self.movie_indices = pd.Series(content_df.index, index=content_df['movieId'])
        
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(content_df['content'])
        
        # Compute Similarity (This might be large, but for 9k movies it's ~9k*9k floats ~ 300MB, acceptable)
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
    def recommend(self, user_id, user_rated_movie_ids, n=10):
        # Recommend items similar to what user liked
        # Simple approach: Average similarity profile of user's liked items
        
        if not user_rated_movie_ids:
            return []
            
        # Get indices of movies user rated
        valid_indices = [self.movie_indices[mid] for mid in user_rated_movie_ids if mid in self.movie_indices]
        
        if not valid_indices:
            return []
            
        # Average similarity vector for this user
        user_sim_scores = self.cosine_sim[valid_indices].mean(axis=0)
        
        # Sort
        sim_scores = list(enumerate(user_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding already rated ideally, but let's just return top)
        top_indices = [i[0] for i in sim_scores[:n]]
        
        return self.movie_indices.index[self.movie_indices.isin(top_indices)].tolist() # Mapping back might be tricky with Series

        # Better mapping
        idx_to_movie = self.movie_indices.index
        return [idx_to_movie[i] for i in top_indices]

class HybridRecommender(BaseRecommender):
    def __init__(self, svd_model, content_model, alpha=0.3):
        self.svd = svd_model
        self.content = content_model
        self.alpha = alpha
        self.user_history = {} # Store train history: userId -> [movieId]
        
    def fit(self, train_df, movies_df, tags_df):
        print("Training Hybrid: Fitting SVD...")
        self.svd.fit(train_df)
        print("Training Hybrid: Fitting Content...")
        self.content.fit(train_df, movies_df, tags_df)
        
        # Store user history for Content-Based Profile generation
        print("Caching User History...")
        self.user_history = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        
    def recommend(self, user_id, n=10):
        # 1. Get SVD Candidates (Top 50)
        # We use SVD as the primary retrieval because scoring all items with Content is expensive per request
        if user_id not in self.svd.user_map:
            return []
            
        u_idx = self.svd.user_map[user_id]
        user_vector = self.svd.user_factors[u_idx]
        svd_item_indices = np.arange(len(self.svd.item_factors))
        svd_scores = np.dot(user_vector, self.svd.item_factors.T)
        
        # Normalize SVD scores (0-1)
        min_s, max_s = svd_scores.min(), svd_scores.max()
        if max_s - min_s > 0:
            svd_scores = (svd_scores - min_s) / (max_s - min_s)
        
        # 2. Content Scores for specific users history
        # We only compute valid item scores? No, we need score for all items to blend?
        # To make it fast, let's only re-rank the top 100 SVD results.
        
        top_indices = np.argsort(svd_scores)[::-1][:100]
        top_movie_ids = [self.svd.reverse_item_map[i] for i in top_indices]
        
        # Calculate Content Score for these 100 movies
        # User profile = Mean of TF-IDF vectors of liked movies
        user_liked_movies = self.user_history.get(user_id, [])
        if not user_liked_movies:
            return self.svd.recommend(user_id, n) # Fallback
            
        # Get indices in Content Matrix
        liked_indices = [self.content.movie_indices[mid] for mid in user_liked_movies if mid in self.content.movie_indices]
        if not liked_indices:
             return self.svd.recommend(user_id, n)

        # User vector
        user_content_profile = np.asarray(self.content.tfidf_matrix[liked_indices].mean(axis=0))
        
        # Candidate vectors
        # Need to Map movie_id -> content_matrix index
        cand_content_indices = [self.content.movie_indices[mid] for mid in top_movie_ids if mid in self.content.movie_indices]
        
        if not cand_content_indices:
             return self.svd.recommend(user_id, n)
             
        # Compute Cosine Similarity between User Profile and Candidates
        # linear_kernel(X, Y) -> X * Y.T
        # user_content_profile is (1, Dim). Candidate_matrix is (100, Dim).
        from sklearn.metrics.pairwise import linear_kernel
        
        candidate_matrix = self.content.tfidf_matrix[cand_content_indices]
        content_scores = linear_kernel(user_content_profile, candidate_matrix).flatten()
        
        # 3. Blend Scores
        # We have SVD scores for these indices.
        # We need to match them up.
        
        # Create a dataframe for easy merging
        candidates_df = pd.DataFrame({
            'movieId': top_movie_ids,
            'svd_score': svd_scores[top_indices]
        })
        
        # Map content scores back
        # Note: cand_content_indices might be smaller if some movies missing from content df
        
        # Create map: movie_id -> content_score
        # candidates in top_movie_ids correspond to cand_content_indices roughly
        # BUT we did a filter `if mid in self.content.movie_indices`.
        # So we need to be careful.
        
        c_scores_map = {}
        valid_c_count = 0
        for mid in top_movie_ids:
            if mid in self.content.movie_indices:
                # content_scores array corresponds to the order of cand_content_indices
                # which corresponds to the filtered order of top_movie_ids
                c_scores_map[mid] = content_scores[valid_c_count]
                valid_c_count += 1
            else:
                c_scores_map[mid] = 0.0
                
        candidates_df['content_score'] = candidates_df['movieId'].map(c_scores_map)
        
        # Final Score
        candidates_df['final_score'] = (1 - self.alpha) * candidates_df['svd_score'] + self.alpha * candidates_df['content_score']
        
        # Sort
        recs = candidates_df.sort_values('final_score', ascending=False).head(n)
        return recs['movieId'].tolist()

class NeuralCFRecommender(BaseRecommender):
    def __init__(self, embedding_dim=32, n_epochs=5, batch_size=256, lr=0.001):
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        
    def fit(self, train_df):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        
        # Mappings
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        # Prepare Dataset
        class RatingDataset(Dataset):
            def __init__(self, df, user_map, item_map):
                self.users = torch.tensor(df['userId'].map(user_map).values, dtype=torch.long)
                self.items = torch.tensor(df['movieId'].map(item_map).values, dtype=torch.long)
                # Normalize ratings to 0-1 for Sigmoid output? Or Regression?
                # Let's use Regression (MSE Loss) for direct rating prediction
                self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
                
            def __len__(self):
                return len(self.ratings)
                
            def __getitem__(self, idx):
                return self.users[idx], self.items[idx], self.ratings[idx]
                
        # Define Model: NeuMF (Neural Matrix Factorization)
        class NeuMF(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim):
                super(NeuMF, self).__init__()
                
                # GMF Part
                self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
                self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)
                
                # MLP Part
                self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
                self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)
                
                self.fc1 = nn.Linear(embedding_dim * 2, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                
                # Concat
                self.output = nn.Linear(embedding_dim + 16, 1)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, user, item):
                # GMF
                u_gmf = self.gmf_user_embedding(user)
                i_gmf = self.gmf_item_embedding(item)
                x_gmf = u_gmf * i_gmf
                
                # MLP
                u_mlp = self.mlp_user_embedding(user)
                i_mlp = self.mlp_item_embedding(item)
                x_mlp = torch.cat([u_mlp, i_mlp], dim=1)
                x_mlp = torch.relu(self.fc1(x_mlp))
                x_mlp = self.dropout(x_mlp)
                x_mlp = torch.relu(self.fc2(x_mlp))
                x_mlp = self.dropout(x_mlp)
                x_mlp = torch.relu(self.fc3(x_mlp))
                
                # Concat
                x = torch.cat([x_gmf, x_mlp], dim=1)
                x = self.output(x)
                return x.squeeze()
        
        # Training
        n_users = len(users)
        n_items = len(items)
        
        dataset = RatingDataset(train_df, self.user_map, self.item_map)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = NeuMF(n_users, n_items, self.embedding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.n_epochs = 20 # Increase epochs
        print(f"Training Neural CF for {self.n_epochs} epochs...")
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for u, i, r in loader:
                optimizer.zero_grad()
                pred = self.model(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len(loader):.4f}")
            
    def recommend(self, user_id, n=10):
        import torch
        
        if self.model is None:
            return []
            
        if user_id not in self.user_map:
            return []
            
        u_idx = self.user_map[user_id]
        
        # Score all items (efficient enough for <10k items)
        self.model.eval()
        with torch.no_grad():
            # Create vectors for inference
            all_item_indices = torch.arange(len(self.item_map), dtype=torch.long)
            user_indices = torch.full((len(self.item_map),), u_idx, dtype=torch.long)
            
            predictions = self.model(user_indices, all_item_indices)
            
            # Top N
            _, top_indices_tensor = torch.topk(predictions, n)
            top_indices = top_indices_tensor.numpy()
            
            return [self.reverse_item_map.get(i) for i in top_indices]

class DiversityRecommender(BaseRecommender):
    def __init__(self, base_recommender, lambda_param=0.5):
        self.base_recommender = base_recommender
        self.lambda_param = lambda_param # 1.0 = Pure Accuracy, 0.0 = Pure Diversity
        self.item_genres = {}
        
    def fit(self, train_df, movies_df, **kwargs):
        # Fit base recommender with any extra args it needs (e.g. tags_df for Hybrid)
        self.base_recommender.fit(train_df, movies_df, **kwargs)
        # Cache item genres for diversity calculation
        # Map movieId -> set of genres
        for idx, row in movies_df.iterrows():
            self.item_genres[row['movieId']] = set(row['genres'].split('|'))
            
    def recommend(self, user_id, n=10):
        # Get more candidates from base model strategy
        candidates = self.base_recommender.recommend(user_id, n=n*3)
        if not candidates:
            return []
            
        # MMR Selection
        selected = []
        candidates_set = set(candidates)
        
        while len(selected) < n and candidates_set:
            best_item = None
            best_score = -float('inf')
            
            for item in candidates_set:
                # Relevance: We assume rank implies relevance score (simple rank-based score)
                # Ideally we want exact scores, but rank proxy is fine: Score = 1/Rank
                rank = candidates.index(item) + 1
                relevance = 1.0 / rank
                
                # Diversity: Max sim to already selected items
                max_sim = 0.0
                if selected:
                    item_genes = self.item_genres.get(item, set())
                    for s_item in selected:
                        s_genres = self.item_genres.get(s_item, set())
                        # Jaccard Sim of genres
                        u = len(item_genes.union(s_genres))
                        i = len(item_genes.intersection(s_genres))
                        sim = i / u if u > 0 else 0
                        max_sim = max(max_sim, sim)
                
                # MMR Score = Lambda * Rel - (1-Lambda) * MaxSim
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item
            
            if best_item:
                selected.append(best_item)
                candidates_set.remove(best_item)
                
        return selected

class EnsembleRecommender(BaseRecommender):
    def __init__(self, models_dict, weights=None):
        self.models = models_dict # dict of name: model
        self.weights = weights if weights else {k: 1.0/len(models_dict) for k in models_dict}
        
    def fit(self, train_df, *args, **kwargs):
        # Models should already be fit generally, but if not we could trigger it.
        # We assume models are pre-fit or fit externally in the pipeline.
        # If we need to fit them:
        for name, model in self.models.items():
            # Check if model has fit method and call it. 
            # This is tricky with different signatures.
            # Ideally passthrough. For now assume external fitting.
            pass
            
    def recommend(self, user_id, n=10):
        # We need scores, not just top-K. 
        # Most recommend methods return just IDs.
        # We need to hack this to get scores or candidate sets.
        
        # Strategy: Get Top-3N from each model, create a candidate pool.
        # Re-score candidate pool using all models (if they support scoring).
        # SVD and LightGCN support scoring. Content supports scoring.
        
        candidates = set()
        for name, model in self.models.items():
            recs = model.recommend(user_id, n=n*3)
            candidates.update(recs)
            
        candidate_list = list(candidates)
        if not candidate_list:
            return []
            
        # Score Map: ItemID -> Weighted Score
        final_scores = {mid: 0.0 for mid in candidate_list}
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            
            # Hacky scoring retrieval. 
            # Ideally each model class has `predict_score(u, i)`.
            # We will do a batch recommend trick or rely on rank.
            
            # Rank-based Scoring (Borda Count-ish)
            # 1/(rank)
            recs = model.recommend(user_id, n=len(candidate_list)*2) # Get huge list
            
            for rank, mid in enumerate(recs):
                if mid in final_scores:
                    # Score = 1 / (rank + 1)
                    score = 1.0 / (rank + 1)
                    final_scores[mid] += score * weight
                    
        # Sort
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_candidates[:n]]
 

class LightGCNRecommender(BaseRecommender):
    def __init__(self, embedding_dim=64, n_layers=3, n_epochs=20, batch_size=1024, lr=0.001):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.adj_matrix = None
        
    def fit(self, train_df):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import scipy.sparse as sp
        
        # Mappings
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        n_users = len(users)
        n_items = len(items)
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        # 1. Build Adjacency Matrix (Graph Construction)
        # R is (n_users, n_items)
        user_idx = train_df['userId'].map(self.user_map).values
        item_idx = train_df['movieId'].map(self.item_map).values
        
        # Interaction matrix R
        R = sp.coo_matrix((np.ones(len(user_idx)), (user_idx, item_idx)), shape=(n_users, n_items))
        
        # Adjacency Matrix A = [0, R; R.T, 0]
        # Shape (n_users + n_items, n_users + n_items)
        top_left = sp.csr_matrix((n_users, n_users))
        bottom_right = sp.csr_matrix((n_items, n_items))
        
        adj_mat = sp.vstack([sp.hstack([top_left, R]), sp.hstack([R.T, bottom_right])])
        
        # Normalize A: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        # Convert to PyTorch Sparse Tensor
        coo = norm_adj.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, shape)
        
        # 2. Define LightGCN Model
        class LightGCN(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim, n_layers, adj_matrix):
                super(LightGCN, self).__init__()
                self.n_users = n_users
                self.n_items = n_items
                self.embedding_dim = embedding_dim
                self.n_layers = n_layers
                self.adj_matrix = adj_matrix
                
                # Initial Embeddings (0-th layer)
                self.user_embedding = nn.Embedding(n_users, embedding_dim)
                self.item_embedding = nn.Embedding(n_items, embedding_dim)
                
                # Init weights
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                
            def forward(self):
                # Graph Propagation
                all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
                embs = [all_emb]
                
                for i in range(self.n_layers):
                    # E^(k+1) = A * E^k
                    all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
                    embs.append(all_emb)
                    
                # Final Embedding = Mean of all layers
                embs = torch.stack(embs, dim=1)
                final_embs = torch.mean(embs, dim=1)
                
                users_emb, items_emb = torch.split(final_embs, [self.n_users, self.n_items])
                return users_emb, items_emb
                
            def get_rating(self, user_indices, item_indices, u_emb, i_emb):
                return (u_emb[user_indices] * i_emb[item_indices]).sum(1)
        
        # 3. Training Loop (MSE for Stability on Small Data)
        # While BPR is standard for ranking, on small sparse datasets like this, 
        # pointwise MSE often stabilizes faster and acts as a robust proxy.
        self.model = LightGCN(n_users, n_items, self.embedding_dim, self.n_layers, self.adj_matrix)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        train_data = np.vstack([user_idx, item_idx]).T
        
        # 30 Epochs: The "Goldilocks" zone for this dataset
        self.n_epochs = 30 
        print(f"Training LightGCN (God Mode - Optimized) for {self.n_epochs} epochs...")
        
        self.model.train()
        for epoch in range(self.n_epochs):
            # Sample negatives
            neg_items = np.random.randint(0, n_items, size=len(train_data))
            
            u_batch = torch.LongTensor(train_data[:, 0])
            pos_i_batch = torch.LongTensor(train_data[:, 1])
            neg_i_batch = torch.LongTensor(neg_items)
            
            optimizer.zero_grad()
            
            # Forward
            u_emb, i_emb = self.model()
            
            # Scores
            pos_scores = (u_emb[u_batch] * i_emb[pos_i_batch]).sum(1)
            neg_scores = (u_emb[u_batch] * i_emb[neg_i_batch]).sum(1)
            
            # MSE Loss: Pos -> 1, Neg -> 0
            pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
            neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.4f}")

    def recommend(self, user_id, n=10):
        import torch
        if self.model is None or user_id not in self.user_map:
            return []
            
        u_idx = self.user_map[user_id]
        
        self.model.eval()
        with torch.no_grad():
            u_emb, i_emb = self.model()
            user_vec = u_emb[u_idx] # (Dim,)
            
            # Score all items
            scores = torch.matmul(i_emb, user_vec) # (N_items,)
            
            _, top_indices = torch.topk(scores, n)
            return [self.reverse_item_map.get(i.item()) for i in top_indices]
            return [self.reverse_item_map.get(i.item()) for i in top_indices]

class SASRecRecommender(BaseRecommender):
    def __init__(self, embedding_dim=64, n_heads=2, n_blocks=2, max_seq_len=50, dropout=0.1, lr=0.001, n_epochs=50, batch_size=128):
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.user_history = {} 
        self.item_genre_matrix = None # (N_items, N_genres)
        
    def fit(self, train_df, movies_df=None, *args, **kwargs):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        
        # Device Config (Restored for Manual Run on User Terminal)
        # Verify: torch.backends.mps.is_available() should be True on Mac M1/M2/M3
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Training SASRec on: {device} (High-Performance Acceleration Active)")
        
        # 1. Prepare Data
        df_sorted = train_df.sort_values(by=['userId', 'timestamp'])
        
        users = df_sorted['userId'].unique()
        items = df_sorted['movieId'].unique()
        n_items = len(items)
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {item: i+1 for i, item in enumerate(items)}
        self.reverse_item_map = {i+1: item for i, item in enumerate(items)}
        self.n_items_vocab = n_items + 1
        
        # --- Context Feature Engineering (Genres) ---
        if movies_df is not None:
            # Create Genre Map
            # movies_df: movieId, title, genres
            all_genres = set()
            for g_str in movies_df['genres']:
                for g in g_str.split('|'):
                    all_genres.add(g)
            
            genre_map = {g: i for i, g in enumerate(list(all_genres))}
            n_genres = len(genre_map)
            
            # Build Item-Genre Matrix (Static Lookup)
            # Row k corresponds to item index k in self.item_map
            # Size: (n_items_vocab, n_genres)
            self.item_genre_matrix = torch.zeros((self.n_items_vocab, n_genres), device=device)
            
            # Fill
            for mid in items: # Only train items
                if mid in self.item_map:
                    idx = self.item_map[mid]
                    # Find genres
                    g_str = movies_df[movies_df['movieId'] == mid]['genres'].values
                    if len(g_str) > 0:
                        gs = g_str[0].split('|')
                        for g in gs:
                            if g in genre_map:
                                self.item_genre_matrix[idx, genre_map[g]] = 1.0
        else:
            n_genres = 0
            
        self.user_history = df_sorted.groupby('userId')['movieId'].apply(list).to_dict()
        
        train_sequences = []
        train_targets = []
        
        for u, history in self.user_history.items():
            seq = [self.item_map[i] for i in history if i in self.item_map]
            for i in range(1, len(seq)):
                t = seq[i]
                s = seq[:i]
                if len(s) > self.max_seq_len:
                    s = s[-self.max_seq_len:]
                pad_len = self.max_seq_len - len(s)
                s_padded = [0] * pad_len + s
                train_sequences.append(s_padded)
                train_targets.append(t)
                
        class SeqDataset(Dataset):
            def __init__(self, seqs, targets):
                self.seqs = torch.LongTensor(seqs)
                self.targets = torch.LongTensor(targets)
            def __len__(self): return len(self.seqs)
            def __getitem__(self, idx): return self.seqs[idx], self.targets[idx]
        
        dataset = SeqDataset(train_sequences, train_targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Context-Aware Transformer
        class ContextSASRec(nn.Module):
            def __init__(self, n_items, hidden_size, n_heads, n_blocks, max_len, dropout, n_genres=0, genre_matrix=None):
                super(ContextSASRec, self).__init__()
                self.item_emb = nn.Embedding(n_items, hidden_size, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, hidden_size)
                self.emb_dropout = nn.Dropout(dropout)
                
                # Context Fusion
                self.use_context = False
                if n_genres > 0 and genre_matrix is not None:
                    self.use_context = True
                    self.genre_matrix = genre_matrix # (N_items, N_genres)
                    self.genre_projection = nn.Linear(n_genres, hidden_size)
                
                # Transformer Layer (Batch First = True for speed)
                encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dropout=dropout, dim_feedforward=hidden_size*4, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
                
                self.layernorm = nn.LayerNorm(hidden_size)
                
            def forward(self, seqs):
                # seqs: (B, L)
                batch_size = seqs.size(0)
                len_seq = seqs.size(1)
                
                # Padding Mask (B, L) - True where should be ignored (0)
                src_key_padding_mask = (seqs == 0)
                
                # 1. Item Embedding
                x = self.item_emb(seqs) # (B, L, H)
                x = x * (self.item_emb.embedding_dim ** 0.5)
                
                # 2. Add Context (Genres)
                if self.use_context:
                    # Look up genres for the sequence items
                    # seqs is indices [B, L]
                    # genre_matrix is [N_vocab, N_genres]
                    # We need [B, L, N_genres]
                    # Embedding lookup works here too if we treat matrix as embedding weights?
                    # But it's multi-hot. 
                    # Correct way: F.embedding(seqs, self.genre_matrix)
                    g_feats = torch.nn.functional.embedding(seqs, self.genre_matrix) # (B, L, N_genres)
                    g_emb = self.genre_projection(g_feats) # (B, L, H)
                    x = x + g_emb # Element-wise sum (Fusion)
                
                # 3. Positional Embedding
                positions = torch.arange(len_seq, device=seqs.device).unsqueeze(0).repeat(batch_size, 1)
                p = self.pos_emb(positions)
                x = x + p
                
                x = self.emb_dropout(x)
                
                # 4. Transformer
                # Causal Mask
                causal_mask = nn.Transformer.generate_square_subsequent_mask(len_seq).to(seqs.device)
                
                # Batch First is True now
                output = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
                output = self.layernorm(output)
                
                return output[:, -1, :] # Last state (B, H)

        self.model = ContextSASRec(self.n_items_vocab, self.embedding_dim, self.n_heads, self.n_blocks, self.max_seq_len, self.dropout, n_genres=n_genres, genre_matrix=self.item_genre_matrix)
        self.model.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training Context-Aware SASRec (MPS Accelerated) for {self.n_epochs} epochs...")
        self.model.train()
        
        for epoch in range(self.n_epochs):
            total_loss = 0
            for seqs, targets in loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                last_state = self.model(seqs) # (B, H)
                
                # Output Layer matches Item Embedding Weight
                # We need to compute logits for ALL items
                # (B, H) @ (N_items, H).T -> (B, N_items)
                logits = torch.matmul(last_state, self.model.item_emb.weight.transpose(0, 1))
                
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len(loader):.4f}")
                
    def recommend(self, user_id, n=10):
        import torch
        device = next(self.model.parameters()).device
        
        history = self.user_history.get(user_id, [])
        if not history: return []
        
        seq = [self.item_map[i] for i in history if i in self.item_map]
        if not seq: return []

        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        seq_padded = [0] * pad_len + seq
        
        seq_tensor = torch.LongTensor([seq_padded]).to(device)
        
        self.model.eval()
        with torch.no_grad():
            last_state = self.model(seq_tensor)
            logits = torch.matmul(last_state, self.model.item_emb.weight.transpose(0, 1))
            
            logits[0, 0] = -float('inf')
            scores, top_indices = torch.topk(logits, n)
            indices = top_indices.cpu().numpy()[0]
            
            return [self.reverse_item_map.get(i) for i in indices]
